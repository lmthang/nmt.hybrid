function [nextState, attnInfo] = rnnStepLayerForward(W_rnn, W_emb, prevState, input, mask, params, isTest, isDecoder, model, trainData)
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

% mask
maskInfo.mask = mask;
maskInfo.unmaskedIds = find(maskInfo.mask);
maskInfo.maskedIds = find(~maskInfo.mask);

% emb input
if isDecoder && params.feedInput
  input_emb = [W_emb(:, input); prevState{end}.softmax_h];
else
  input_emb = W_emb(:, input);
end

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
  x_t(:, maskInfo.maskedIds) = 0; 
  h_t_1(:, maskInfo.maskedIds) = 0;
  c_t_1(:, maskInfo.maskedIds) = 0;

  % core LSTM
  [nextState{ll}] = lstmUnitForward(W_rnn{ll}, x_t, h_t_1, c_t_1, params, isTest, ll==1 && isDecoder && params.feedInput);
  
  % assert
  if params.assert
    assert(computeSum(nextState{ll}.h_t(:, maskInfo.maskedIds), params.isGPU)==0);
    assert(computeSum(nextState{ll}.c_t(:, maskInfo.maskedIds), params.isGPU)==0);
  end
end

% decoder
attnInfo = [];
if isDecoder
  % attention
  if params.attnFunc 
    % TODO: save memory here, attnInfo.input only keeps track of srcHidVecs or attnVecs, but not h_t.
    [attnInfo] = attnLayerForward(nextState{end}.h_t, params, model, trainData, maskInfo);
    nextState{end}.softmax_h = attnInfo.softmax_h;
  else
    nextState{end}.softmax_h = nextState{end}.h_t;
  end
end