function [nextState, attnInfo] = rnnStepLayerForward(W_rnn, input_emb, prevState, mask, params, rnnFlags, attnData, model)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: to feed the very first layer, lstmSize * batchSize
%   isTest: 1 -- don't store intermediate results
%   attnData: only needed when isDecoder=1, has attnData.srcHidVecsOrig and attnData.srcLens
% Output:
%   nextState
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

numLayers = length(W_rnn);
nextState = cell(numLayers, 1);

% mask
if size(mask, 1) > 1 % make sure it's a row vector
  mask = mask';
end
maskInfo.mask = mask;
maskInfo.unmaskedIds = find(maskInfo.mask);
maskInfo.maskedIds = find(~maskInfo.mask);

% emb input
if rnnFlags.feedInput
  input_emb = [input_emb; prevState{end}.softmax_h];
end

for ll=1:numLayers % layer
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
  rnnFlags.feedInput = (ll==1 && rnnFlags.decode && rnnFlags.feedInput);
  [nextState{ll}] = lstmUnitForward(W_rnn{ll}, x_t, h_t_1, c_t_1, params, rnnFlags);
  
  % assert
  if params.assert
    assert(computeSum(nextState{ll}.h_t(:, maskInfo.maskedIds), params.isGPU)==0);
    assert(computeSum(nextState{ll}.c_t(:, maskInfo.maskedIds), params.isGPU)==0);
  end
end

% decoder
attnInfo = [];
if rnnFlags.decode
  % attention
  if rnnFlags.attn 
    % TODO: save memory here, attnInfo.input only keeps track of srcHidVecs or attnVecs, but not h_t.
    [attnInfo] = attnLayerForward(nextState{end}.h_t, params, model, attnData, maskInfo);
    nextState{end}.softmax_h = attnInfo.softmax_h;
  else
    nextState{end}.softmax_h = nextState{end}.h_t;
  end
end