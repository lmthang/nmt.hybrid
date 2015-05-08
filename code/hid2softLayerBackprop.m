function [grad_ht, hid2softGrad, grad_srcHidVecs] = hid2softLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, h_t, softmax_h, isPredictPos, params)
  grad_srcHidVecs = [];
  
  % softmax compression f(W_h * h_t)
  if params.softmaxDim>0 
    [grad_ht, hid2softGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h_t, params.nonlinear_f_prime, softmax_h);
  
  % positional model 3 f(W_h * [srcPosVecs; h_t])
  elseif params.posModel==3 && isPredictPos==0
    [grad_ht, hid2softGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h2sInfo.input, params.nonlinear_f_prime, softmax_h);

    % grad srcPosVecs
    grad_srcHidVecs = grad_ht(1:params.lstmSize, :);

    % grad_ht: this line needs to come after the above line
    grad_ht = grad_ht(params.lstmSize+1:end, :);
  
  % attention model
  elseif params.attnFunc>0 % f(W_ah*[attn_t; tgt_h_t])
    % softmax_h = f(W_ah*[attn_t; tgt_h_t])  
    [grad_ah, hid2softGrad.W_ah] = hiddenLayerBackprop(model.W_ah, grad_softmax_h, h2sInfo.attn_h_concat, params.nonlinear_f_prime, softmax_h);

    if params.attnRelativePos % relative
      srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
      srcHidVecs(:, :, h2sInfo.startHidId:h2sInfo.endHidId) = trainData.srcHidVecs(:, :, h2sInfo.startAttnId:h2sInfo.endAttnId);
    else
      srcHidVecs = trainData.absSrcHidVecs;
    end
    
    % grad_attn -> grad_ht, grad_W_a, grad_srcHidVecs
    [grad_ht, hid2softGrad.W_a, grad_srcHidVecs] = attnLayerBackprop(model.W_a, grad_ah(1:params.lstmSize, :), h_t, params, ...
      h2sInfo.alignWeights, srcHidVecs);

    % grad_ht
    grad_ht = grad_ht + grad_ah(params.lstmSize+1:end, :);
  end
end