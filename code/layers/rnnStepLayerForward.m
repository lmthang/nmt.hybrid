function [lstm, h_t, c_t] = rnnStepLayerForward(W_layers, prevState, input, maskedIds, params)
% Stacking RNN at one time step
% Input:
%   W_layers: recurrent connections of multiple layers, e.g., W_layers{ll}.
%   prevState: previous hidden state, e.g., for LSTM, prevState.c{ll}, prevState.h{ll}.
%   input: to feed the very first layer, lstmSize * batchSize
%   isTest: 1 -- don't store intermediate results
%
% Output:
%   lstm struct
% Thang Luong @ 2014, <lmthang@stanford.edu>

nextState = cell(params.numLayers, 1);
for ll=1:params.numLayers % layer
  W = W_layers{ll};

  % previous state
  h_t_1 = prevState.h;
  c_t_1 = prevState.c;
  
  % input
  if ll==1 % first layer
    x_t = input;
  else % subsequent layer, use the previous-layer hidden state
    x_t = nextState.h{ll-1};
  end

  % masking
  x_t(:, maskedIds) = 0; 
  h_t_1(:, maskedIds) = 0;
  c_t_1(:, maskedIds) = 0;

  %% Core LSTM: input -> h_t
  [lstms{ll, tt}, h_t{ll}, all_c_t{ll, tt}] = lstmLayerForward(W, x_t, h_t_1, c_t_1, ll, tt, srcMaxLen, params, isTest); 
  % assert
  if params.assert
    assert(computeSum(h_t{ll}(:, curMask.maskedIds), params.isGPU)==0);
  end
end