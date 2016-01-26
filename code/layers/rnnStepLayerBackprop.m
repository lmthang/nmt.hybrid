function [dc, dh, d_emb, grad_W_rnn, d_feed_input] = rnnStepLayerBackprop(W_rnn, prev_state, cur_state, cur_top_grad, dc, dh, ...
  maskInfo, params, isFeedInput)
% Backprop multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   curState: current hidden state, e.g., for LSTM, curState.c{ll}, curState.h{ll}.
%
% Output:
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

numLayers = length(W_rnn);

unmaskedIds = maskInfo.unmaskedIds;
maskedIds = maskInfo.maskedIds;
grad_W_rnn = cell(numLayers, 1);
for ll=numLayers:-1:1 % layer
  % add grad from the top
  if ~isempty(cur_top_grad)
    dh{ll}(:, unmaskedIds) = dh{ll}(:, unmaskedIds) + cur_top_grad(1:params.lstmSize, unmaskedIds);
  end

  % cell backprop
  [dc{ll}, dh{ll}, d_input, grad_W_rnn{ll}] = lstmUnitBackprop(W_rnn{ll}, cur_state{ll}, prev_state{ll}.c_t, dc{ll}, dh{ll}, maskedIds, ...
    params, ll==1 && isFeedInput);

  % pass down hidden state grad to the below layer 
  cur_top_grad = d_input;
end

% emb
d_emb = d_input(1:params.lstmSize, :);

% feed softmax vector
if isFeedInput
  d_feed_input = d_input(params.lstmSize+1:2*params.lstmSize, :);
else
  d_feed_input = [];
end