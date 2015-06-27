%%%
%
% Attentional Layer Backprop from softmax hidden state to lstm hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, params, curMask) %isPredictPos, params)
  % softmax_h -> h_t
  [grad_input, attnGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h2sInfo.input, params.nonlinear_f_prime, h2sInfo.softmax_h);

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

  % grad_contextVecs -> grad_srcHidVecs, grad_alignWeights
  grad_contextVecs = grad_input(1:params.lstmSize, :);
  [grad_alignWeights, grad_srcHidVecs] = contextLayerBackprop(grad_contextVecs, h2sInfo.alignWeights, srcHidVecs, params);

  % grad_contextVecs -> grad_ht, grad_W_a, grad_srcHidVecs
  if params.predictPos==3
    % IMPORTANT: don't change the order of these lines
    % grad_alignWeights -> grad_distWeights, grad_preAlignWeights
    grad_distWeights = grad_alignWeights.*h2sInfo.preAlignWeights;
    grad_alignWeights = grad_alignWeights.*h2sInfo.distWeights; % grad_preAlignWeights
    h2sInfo.alignWeights = h2sInfo.preAlignWeights;
  end
  
  % grad_scores -> grad_ht, grad_W_a / grad_srcHidVecs
  if params.attnOpt==0 % no src compare
    % grad_alignWeights -> grad_scores
    [grad_scores] = normLayerBackprop(grad_alignWeights, h2sInfo.alignWeights, h2sInfo.maskedIds, params);
    
    % s_t = W_a * h_t
    [grad_ht, attnGrad.W_a] = linearLayerBackprop(model.W_a, grad_scores, h2sInfo.h_t);  
    
    % assert
    if params.assert
      assert(computeSum(grad_scores(h2sInfo.maskedIds), params.isGPU)==0);
      assert(computeSum(grad_ht(:, curMask.maskedIds), params.isGPU)==0);
    end
    
  elseif params.attnOpt==1 || params.attnOpt==2
    if params.attnOpt==1 % s_t = H_src * h_t
      [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_alignWeights, h2sInfo.alignWeights, srcHidVecs, h2sInfo.h_t, h2sInfo.maskedIds, params);
    elseif params.attnOpt==2 % s_t = H_src * W_a * h_t
      [grad_transform_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_alignWeights, h2sInfo.alignWeights, srcHidVecs, h2sInfo.transform_ht, h2sInfo.maskedIds, params);
      [grad_ht, attnGrad.W_a] = linearLayerBackprop(model.W_a, grad_transform_ht, h2sInfo.h_t);
    end
    
    grad_srcHidVecs = grad_srcHidVecs + grad_srcHidVecs1; % add to the existing grad_srcHidVecs
  end

  % grad_ht
  grad_ht = grad_ht + grad_input(params.lstmSize+1:end, :);
  
  % assert
  if params.assert
    assert(computeSum(grad_input(params.lstmSize+1:end, curMask.maskedIds), params.isGPU)==0);
  end
  
  if params.predictPos==3
    % since linearIdSub is for matrix of size [curBatchSize, numAttnPositions], 
    % we need to transpose grad_alignWeights to be of that size.
    grad_distWeights = grad_distWeights';
    h2sInfo.distWeights = h2sInfo.distWeights';

    % grad_distWeights -> grad_mu
    [grad_mu] = distLayerBackprop(grad_distWeights, h2sInfo.distWeights, h2sInfo, params);

    % grad_mu -> grad_scales
    grad_scales = trainData.srcLens.*grad_mu;

    % grad_scales -> grad_ht, grad_W_pos, grad_v_pos
    [grad_ht1, attnGrad.W_pos, attnGrad.v_pos] = scaleLayerBackprop(model.W_pos, model.v_pos, grad_scales, h2sInfo.h_t, h2sInfo.scales, h2sInfo.posForwData, params);
    grad_ht = grad_ht + grad_ht1;
  end
end

%     if params.attnOpt==1
%       % grad_compareWeights -> grad_scores
%       [grad_scores] = normLayerBackprop(grad_preAlignWeights, h2sInfo.compareWeights, params);
% 
%       % grad_scores -> grad_ht, grad_srcHidVecs
%       [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_scores, srcHidVecs, h2sInfo.h_t);
%       grad_srcHidVecs = grad_srcHidVecs + grad_srcHidVecs1; % add to the existing grad_srcHidVecs
%     end


%         if params.attnOpt==0 
%           % since linearIdSub is for matrix of size [curBatchSize, numAttnPositions], 
%           % we need to transpose grad_alignWeights to be of that size.
%           grad_alignWeights = grad_alignWeights';
%           h2sInfo.alignWeights = h2sInfo.alignWeights';
%           
%           % grad_alignWeights -> grad_variances, grad_mu
%           [grad_mu, grad_variances] = gaussLayerBackprop(grad_alignWeights, h2sInfo, params);
% 
%           % grad_variances -> grad_h_t, grad_W_var, grad_v_var, scales=sigmoid(v_pos*f(W_pos*h_t)) in [0, 1]
%           [grad_ht, hid2softGrad.W_var, hid2softGrad.v_var] = scaleLayerBackprop(model.W_var, model.v_var, grad_variances, h2sInfo.h_t, ...
%             h2sInfo.origVariances, h2sInfo.varForwData, params);
%         else
%         end

%     params.softmaxDim || 
%     if params.softmaxDim % softmax compression: f(W_h * h_t)
%       grad_ht = grad_input;
%     elseif params.attnFunc>0 % attention model f(W_h*[attn_t; tgt_h_t])      
%       
%     end

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
%         % grad_contextVecs
%         grad_contextVecs = grad_input(1:params.lstmSize, :);
%         
%         % grad srcHidVecs
%         % attn_t = alignWeights*srcHidVecs;
%         grad_srcHidVecs = bsxfun(@times, h2sInfo.alignWeights, grad_contextVecs); % alignWeights*grad_contextVecs
%         
%         % grad_align_weights = srcHidVecs' * grad_contextVecs
%         srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
%         srcHidVecs(:, h2sInfo.unmaskedIds) = reshape(trainData.srcHidVecs(h2sInfo.linearIndices), params.lstmSize, length(h2sInfo.unmaskedIds)); 
%         grad_alignWeights = sum(srcHidVecs.*grad_contextVecs);
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