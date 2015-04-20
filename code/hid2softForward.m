function [softmax_h, hid2softData] = hid2softForward(h_t, params, model, batchData, mask, isPredictPos)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%  
  
  hid2softData.input = h_t;
  
  if params.attnFunc>0 % attention mechanism
    % softmax_h = f(W_ah*[attn_t; tgt_h_t])
    [attnVecs, hid2softData.alignWeights] = attnLayerForward(model, h_t, batchData.srcHidVecs, mask);
    hid2softData.attn_h_concat = [attnVecs; h_t];
    [softmax_h] = hiddenLayerForward(model.W_ah, hid2softData.attn_h_concat, params.nonlinear_f);
  elseif params.softmaxDim>0 || (params.posModel==3 && isPredictPos==0) % compression: f(W_h * h_t) or positional model: f(W_h * [srcPosVecs; h_t])
    if params.posModel==3
      hid2softData.input = [batchData.srcPosVecs; h_t];
    end
    
    softmax_h = hiddenLayerForward(model.W_h, hid2softData.input, params.nonlinear_f);
  else
    softmax_h = h_t;
  end
end
    