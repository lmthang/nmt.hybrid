function [scales, forwardData] = scaleLayerForward(W, v, h_t, params)
%%%
%
% From lstm hidden states h_t -> scales in [0, 1].
%   scales=sigmoid(v*f(W*h_t))
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 

  % h_t -> h_pos=f(W*h_t)
  forwardData.h_pos = hiddenLayerForward(W, h_t, params.nonlinear_f);
  
  % h_pos -> scales=sigmoid(v*h_pos) in [0, 1]
  scales = hiddenLayerForward(v, forwardData.h_pos, params.nonlinear_gate_f);
end

%   if params.absolutePos % 
%     positions = trainData.srcLens.*forwardData.scales;  
%   else % scales=tanh(v*h_pos) in [-1, 1]
%     forwardData.scales = hiddenLayerForward(v, forwardData.h_pos, params.nonlinear_f);
%     positions = params.posWin.*forwardData.scales;
%   end
