function [nextState] = rnnStepLayerForward(W_rnn, prevState, input_emb, maskedIds, params, isTest)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: to feed the very first layer, lstmSize * batchSize
%   isTest: 1 -- don't store intermediate results
%
% Output:
%   nextState
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

nextState = cell(params.numLayers, 1);

for ll=1:params.numLayers % layer
  % previous state
  h_t_1 = prevState{ll}.h_t;
  c_t_1 = prevState{ll}.c_t;
  
  % input
  if ll==1 % first layer
    x_t = input_emb;
  else % subsequent layer, use the previous-layer hidden state
    x_t = nextState{ll-1}.h_t;
  end

  % masking
  x_t(:, maskedIds) = 0; 
  h_t_1(:, maskedIds) = 0;
  c_t_1(:, maskedIds) = 0;

  % core LSTM
  [nextState{ll}] = lstmLayerForward(W_rnn{ll}, x_t, h_t_1, c_t_1, params, isTest); % % ll, tt, srcMaxLen, 
  
  % assert
  if params.assert
    assert(computeSum(nextState{ll}.h_t(:, maskedIds), params.isGPU)==0);
    assert(computeSum(nextState{ll}.c_t(:, maskedIds), params.isGPU)==0);
  end
end