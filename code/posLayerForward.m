function [scales, forwardData] = posLayerForward(W_pos, v_pos, h_t, params)
%%%
%
% From lstm hidden states h_t -> predicted relative positions.
%   scales=g(v_pos*f(W_pos*h_t)), where g=sigmoid.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 

  % h_t -> h_pos=f(W_pos*h_t)
  forwardData.h_pos = hiddenLayerForward(W_pos, h_t, params.nonlinear_f);
  
  % h_pos -> scales=sigmoid(v_pos*h_pos) in [0, 1]
  scales = hiddenLayerForward(v_pos, forwardData.h_pos, params.nonlinear_gate_f);
end

%   if params.absolutePos % 
%     positions = trainData.srcLens.*forwardData.scales;  
%   else % scales=tanh(v_pos*h_pos) in [-1, 1]
%     forwardData.scales = hiddenLayerForward(v_pos, forwardData.h_pos, params.nonlinear_f);
%     positions = params.posWin.*forwardData.scales;
%   end
