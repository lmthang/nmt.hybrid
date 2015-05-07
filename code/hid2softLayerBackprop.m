function [grad_ht, hid2softGrad, grad_srcHidVecs] = hid2softLayerBackprop(model, grad_softmax_h, hid2softData, softmax_h, isPredictPos, params)
  grad_srcHidVecs = [];
  if params.softmaxDim>0 || (params.posModel==3 && isPredictPos==0) % softmax compression or attention or posModel 3
    % f(W_h * input)
    [grad_ht, hid2softGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, hid2softData.input, params.nonlinear_f_prime, softmax_h);

    if params.posModel==3 % input = [srcPosVecs; h_t]
      % grad srcPosVecs
      grad_srcHidVecs = grad_ht(1:params.lstmSize, :);

      % grad_ht: this line needs to come after the above line
      grad_ht = grad_ht(params.lstmSize+1:end, :);
    end
  elseif params.attnFunc>0 % f(W_ah*[attn_t; tgt_h_t])
    % softmax_h = f(W_ah*[attn_t; tgt_h_t])  
    [grad_ah, hid2softGrad.W_ah] = hiddenLayerBackprop(model.W_ah, grad_softmax_h, hid2softData.attn_h_concat, params.nonlinear_f_prime, softmax_h);

    % grad_attn -> grad_ht, grad_W_a, grad_srcHidVecs
    [grad_ht, hid2softGrad.W_a, grad_srcHidVecs] = attnLayerBackprop(model.W_a, grad_ah(1:params.lstmSize, :), hid2softData.input, params, hid2softData.alignWeights, hid2softData.srcHidVecs);

    % grad_ht
    grad_ht = grad_ht + grad_ah(params.lstmSize+1:end, :);
  end
end