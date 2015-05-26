%%%
%
% Backprop from softmax hidden state to lstm hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, hid2softGrad, grad_srcHidVecs] = hid2softLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, params) %isPredictPos, params)
  grad_srcHidVecs = [];
  
  if params.softmaxDim || params.attnFunc %|| (params.posModel==3 && isPredictPos==0)
    % softmax_h -> h_t
    [grad_input, hid2softGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h2sInfo.input, params.nonlinear_f_prime, h2sInfo.softmax_h);
    
    if params.softmaxDim % softmax compression: f(W_h * h_t)
      grad_ht = grad_input;
    elseif params.attnFunc>0 % attention model f(W_h*[attn_t; tgt_h_t])      
      if params.attnGlobal % soft attention
        srcHidVecs = trainData.absSrcHidVecs;
      else % hard attention
        % TODO: if our GPUs have lots of memory, then we don't have to
        % regenerate srcHidVecs again :) Unfortuntely not!
        if ~isempty(h2sInfo.linearIdSub)
          srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize*params.numAttnPositions], params.isGPU, params.dataType);
          trainData.srcHidVecs = reshape(trainData.srcHidVecs, params.lstmSize, []);
          srcHidVecs(:, h2sInfo.linearIdSub) = trainData.srcHidVecs(:, h2sInfo.linearIdAll);
          srcHidVecs = reshape(srcHidVecs, [params.lstmSize, params.curBatchSize, params.numAttnPositions]);
        else
          srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
        end
      end
      
      % grad_attn -> grad_ht, grad_W_a, grad_srcHidVecs
      if params.predictPos==3
        % grad_attn -> grad_srcHidVecs, grad_alignWeights
        outGrad = permute(grad_input(1:params.lstmSize, :), [1, 2, 3]); % change from lstmSize*curBatchSize -> lstmSize*curBatchSize*1
        grad_srcHidVecs = bsxfun(@times, outGrad, h2sInfo.alignWeights); 
        if params.numAttnPositions==1
          grad_alignWeights = sum(srcHidVecs.*outGrad); % sum across lstmSize
        else
          grad_alignWeights = squeeze(sum(bsxfun(@times, srcHidVecs, outGrad), 1))'; % bsxfun along numAttnPositions, sum across lstmSize
        end
        
        grad_alignWeights = grad_alignWeights';
        h2sInfo.alignWeights = squeeze(h2sInfo.alignWeights);
        if params.assert
          assert(size(grad_alignWeights, 1)==trainData.curBatchSize);
          assert(size(grad_alignWeights, 2)==params.numAttnPositions);
          assert(size(h2sInfo.alignWeights, 1)==trainData.curBatchSize);
          assert(size(h2sInfo.alignWeights, 2)==params.numAttnPositions);
        end
        
        % grad_alignWeights -> grad_variances
        % 0.5*p*(scaleX^2/variance - 1/sigAbs)
        if params.isGPU
          grad_variances = arrayfun(@gradSigSquare, grad_alignWeights(h2sInfo.linearIdSub), ...
            h2sInfo.alignWeights(h2sInfo.linearIdSub), h2sInfo.scaledPositions, h2sInfo.variances, h2sInfo.sigAbs);
        else
          grad_variances = 0.5*grad_alignWeights(h2sInfo.linearIdSub).*h2sInfo.alignWeights(h2sInfo.linearIdSub).*...
            (h2sInfo.scaledPositions.^2./h2sInfo.variances-1./h2sInfo.sigAbs);
          %assert(sum(sum(abs(grad_variances-grad_variances1)))==0);
        end
        
        % grad_alignWeights -> grad_mu
        % 0.5*p*(scaleX^2/variance - 1/sigAbs)
        if params.isGPU
          grad_mu = arrayfun(@gradMu, grad_alignWeights(h2sInfo.linearIdSub), h2sInfo.alignWeights(h2sInfo.linearIdSub), ...
            h2sInfo.scaledPositions, h2sInfo.sigAbs);
        else
          grad_mu = grad_alignWeights(h2sInfo.linearIdSub).*h2sInfo.alignWeights(h2sInfo.linearIdSub).*h2sInfo.scaledPositions./h2sInfo.sigAbs;
          %assert(sum(sum(abs(grad_mu-grad_mu1)))==0);
        end
        
        % accumulate grad_variances, grad_mu
        [grad_variances_accum, indices_variances] = aggregateMatrix(grad_variances, h2sInfo.unmaskedIds, params.isGPU, params.dataType);
        [grad_mu_accum, indices_mu] = aggregateMatrix(grad_mu, h2sInfo.unmaskedIds, params.isGPU, params.dataType);
        %assert(sum(sum(abs(indices_variances-indices_mu)))==0);
        grad_variances = zeroMatrix([1, trainData.curBatchSize], params.isGPU, params.dataType);
        grad_variances(indices_variances) = grad_variances_accum;
        grad_mu = zeroMatrix([1, trainData.curBatchSize], params.isGPU, params.dataType);
        grad_mu(indices_mu) = grad_mu_accum;
        
        
        % grad_variances -> grad_h_t, grad_W_var, grad_v_var, scales=sigmoid(v_pos*f(W_pos*h_t)) in [0, 1]
        [grad_ht, hid2softGrad.W_var, hid2softGrad.v_var] = posLayerBackprop(model.W_var, model.v_var, grad_variances, h2sInfo.h_t, ...
          h2sInfo.origVariances, h2sInfo.varForwData, params);
        
        % grad_mu -> grad_scales
        grad_scales = trainData.srcLens.*grad_mu;
        
        % grad_scales -> grad_ht, grad_W_pos, grad_v_pos
        [grad_ht1, hid2softGrad.W_pos, hid2softGrad.v_pos] = posLayerBackprop(model.W_pos, model.v_pos, grad_scales, h2sInfo.h_t, h2sInfo.scales, h2sInfo.posForwData, params);
        grad_ht = grad_ht + grad_ht1;
      else
        [grad_ht, hid2softGrad.W_a, grad_srcHidVecs] = attnLayerBackprop(model.W_a, grad_input(1:params.lstmSize, :), h2sInfo.h_t, params, ...
          h2sInfo.alignWeights, srcHidVecs);  
      end
      
      % grad_ht
      grad_ht = grad_ht + grad_input(params.lstmSize+1:end, :);
    end
  end
end

function [grad_variance] = gradSigSquare(grad_align, alignWeight, scaledX, variance, sigAbs)
  grad_variance = 0.5*grad_align*alignWeight*(scaledX^2/variance-1/sigAbs);
end

function [grad_mu] = gradMu(grad_align, alignWeight, scaledX, sigAbs)
  grad_mu = grad_align*alignWeight*scaledX/sigAbs;
end

%         if params.predictPos % use unsupervised alignments
%         else % use monotonic alignments
%           srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
%           srcHidVecs(:, :, h2sInfo.startHidId:h2sInfo.endHidId) = trainData.srcHidVecs(:, :, h2sInfo.startAttnId:h2sInfo.endAttnId);
%         end

%           if params.oldSrcVecs % old
%             srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
%             srcHidVecs(h2sInfo.attnLinearIndices) = trainData.srcHidVecs(h2sInfo.linearIndices);
%           else % new  
%           end          

%     elseif params.posModel==3 && isPredictPos==0 % positional model 3 f(W_h * [srcPosVecs; h_t])
%       % grad srcPosVecs
%       grad_srcHidVecs = grad_input(1:params.lstmSize, :);
% 
%       % grad_ht: this line needs to come after the above line
%       grad_ht = grad_input(params.lstmSize+1:end, :);

%       if params.predictPos % hard attention
%         % grad_attn
%         grad_attn = grad_input(1:params.lstmSize, :);
%         
%         % grad srcHidVecs
%         % attn_t = alignWeights*srcHidVecs;
%         grad_srcHidVecs = bsxfun(@times, h2sInfo.alignWeights, grad_attn); % alignWeights*grad_attn
%         
%         % grad_align_weights = srcHidVecs' * grad_attn
%         srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
%         srcHidVecs(:, h2sInfo.unmaskedIds) = reshape(trainData.srcHidVecs(h2sInfo.linearIndices), params.lstmSize, length(h2sInfo.unmaskedIds)); 
%         grad_alignWeights = sum(srcHidVecs.*grad_attn);
%         
%         % align_weights = sigmoid(align_scores)
%         
%         grad_alignScores = params.nonlinear_gate_f_prime(h2sInfo.alignWeights).*grad_alignWeights;
%         
%         % align_scores = v_pos*h_pos
%         [grad_h_pos, hid2softGrad.v_pos] = linearLayerBackprop(model.v_pos, grad_alignScores, h2sInfo.h_pos);  
%         
%         % h_pos = f(W_h*h_t)
%         % h_pos -> h_t
%         [grad_ht, hid2softGrad.W_h_pos] = hiddenLayerBackprop(model.W_h_pos, grad_h_pos, h2sInfo.h_t, params.nonlinear_f_prime, h2sInfo.h_pos);
%       else % soft attention
%       end