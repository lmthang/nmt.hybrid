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
      
      if params.attnRelativePos
        % TODO: if our GPUs have lots of memory, then we don't have to
        % regenerate srcHidVecs again :) Unfortuntely not!
        if params.predictPos % use unsupervised alignments
          srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize*params.numAttnPositions], params.isGPU, params.dataType);
          trainData.srcHidVecs = reshape(trainData.srcHidVecs, params.lstmSize, []);
          srcHidVecs(:, h2sInfo.linearIdSub) = trainData.srcHidVecs(:, h2sInfo.linearIdAll);
          srcHidVecs = reshape(srcHidVecs, [params.lstmSize, params.curBatchSize, params.numAttnPositions]);
        elseif params.attnRelativePos % relative (approximate aligned src position by tgtPos)
          srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
          srcHidVecs(:, :, h2sInfo.startHidId:h2sInfo.endHidId) = trainData.srcHidVecs(:, :, h2sInfo.startAttnId:h2sInfo.endAttnId);
        end
      else
        srcHidVecs = trainData.absSrcHidVecs;
      end
      % grad_attn -> grad_ht, grad_W_a, grad_srcHidVecs
      [grad_ht, hid2softGrad.W_a, grad_srcHidVecs] = attnLayerBackprop(model.W_a, grad_input(1:params.lstmSize, :), h2sInfo.h_t, params, ...
        h2sInfo.alignWeights, srcHidVecs);  
      
      % grad_ht
      grad_ht = grad_ht + grad_input(params.lstmSize+1:end, :);
    end
  end
end

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