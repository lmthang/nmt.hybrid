function [dc, dh, d_emb, grad_W_rnn, d_feed_input] = rnnStepLayerBackprop(W_rnn, prevState, curState, topHidGrad, dc, dh, maskInfo, params, isDecoder)
% Backprop multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   curState: current hidden state, e.g., for LSTM, curState.c{ll}, curState.h{ll}.
%
% Output:
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

unmaskedIds = maskInfo.unmaskedIds;
maskedIds = maskInfo.maskedIds;
grad_W_rnn = cell(params.numLayers, 1);
for ll=params.numLayers:-1:1 % layer
  % add grad from the top
  dh{ll}(:, unmaskedIds) = dh{ll}(:, unmaskedIds) + topHidGrad(1:params.lstmSize, unmaskedIds);

  % cell backprop
  [dc{ll}, dh{ll}, d_input, grad_W_rnn{ll}] = lstmUnitBackprop(W_rnn{ll}, curState{ll}, prevState{ll}.c_t, dc{ll}, dh{ll}, maskedIds, params);

  % pass down hidden state grad to the below layer 
  topHidGrad = d_input;
end

% emb
d_emb = d_input(1:params.lstmSize, :);

% feed softmax vector
if params.feedInput && isDecoder
  d_feed_input = d_input(params.lstmSize+1:2*params.lstmSize, :);
else
  d_feed_input = [];
end