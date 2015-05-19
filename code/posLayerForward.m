function [positions, forwardData] = posLayerForward(W_pos, v_pos, h_t, params, trainData)
%%%
%
% From lstm hidden states h_t -> predicted positions.
%   pos=len*g(v_pos*f(W_pos*h_t)) 
% For absolute positions, len=srcLen, g=sigmoid.
% For relative positions, len=posWin, g=tanh.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 

  % h_t -> h_pos=f(W_pos*h_t)
  [forwardData.h_pos] = hiddenLayerForward(W_pos, h_t, params.nonlinear_f);

  % h_pos -> positions
  if params.absolutePos % scales=sigmoid(v_pos*h_pos) in [0, 1]
    forwardData.scales = hiddenLayerForward(v_pos, forwardData.h_pos, params.nonlinear_gate_f);
    positions = trainData.srcLens.*forwardData.scales;
  else % scales=tanh(v_pos*h_pos) in [-1, 1]
    forwardData.scales = hiddenLayerForward(v_pos, forwardData.h_pos, params.nonlinear_f);
    positions = params.posWin.*forwardData.scales;
  end
end