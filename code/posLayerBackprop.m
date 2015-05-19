%%%
%
% Backprop from predicted positions to lstm hidden states h_t.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, grad_W_pos, grad_v_pos] = posLayerBackprop(W_pos, v_pos, grad_positions, h_t, forwardData, trainData, params)

  % positions -> h_pos
  if params.absolutePos % scales=sigmoid(v_pos*h_pos) in [0, 1]
    grad_scales = trainData.srcLens.*grad_positions;
    [grad_h_pos, grad_v_pos] = hiddenLayerBackprop(v_pos, grad_scales, forwardData.h_pos, params.nonlinear_gate_f_prime, forwardData.scales);
  else % scales=tanh(v_pos*h_pos) in [-1, 1]
    grad_scales = params.posWin.*grad_positions;
    [grad_h_pos, grad_v_pos] = hiddenLayerBackprop(v_pos, grad_scales, forwardData.h_pos, params.nonlinear_f_prime, forwardData.scales);
  end

  % h_pos -> h_t
  [grad_ht, grad_W_pos] = hiddenLayerBackprop(W_pos, grad_h_pos, h_t, params.nonlinear_f_prime, forwardData.h_pos);
end