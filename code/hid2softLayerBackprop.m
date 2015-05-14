%%%
%
% Backprop from softmax hidden state to lstm hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, hid2softGrad, grad_srcHidVecs] = hid2softLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, isPredictPos, params)
  grad_srcHidVecs = [];
  
  if params.softmaxDim || params.attnFunc || (params.posModel==3 && isPredictPos==0)
    % softmax_h -> h_t
    [grad_input, hid2softGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h2sInfo.input, params.nonlinear_f_prime, h2sInfo.softmax_h);
    
    if params.softmaxDim % softmax compression: f(W_h * h_t)
      grad_ht = grad_input;
    elseif params.posModel==3 && isPredictPos==0 % positional model 3 f(W_h * [srcPosVecs; h_t])
      % grad srcPosVecs
      grad_srcHidVecs = grad_input(1:params.lstmSize, :);

      % grad_ht: this line needs to come after the above line
      grad_ht = grad_input(params.lstmSize+1:end, :);
    elseif params.attnFunc>0 % attention model f(W_h*[attn_t; tgt_h_t])
      if params.attnRelativePos % relative
        srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
        srcHidVecs(:, :, h2sInfo.startHidId:h2sInfo.endHidId) = trainData.srcHidVecs(:, :, h2sInfo.startAttnId:h2sInfo.endAttnId);
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