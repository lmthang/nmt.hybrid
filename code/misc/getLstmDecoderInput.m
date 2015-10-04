function [x_t] = getLstmDecoderInput(wordId, W_emb, softmax_h, params)
% getLstmDecoderInput - prepare input vector to the decoder
%
% Input:
%   wordId: index of the input word
%   W_emb: embedding matrix
%   softmax_h: previous hidden state
%   params: parameter settings
%
% Output:
%   x_t: input vector
%
% Authors: 
%   Thang Luong @ 2015, <lmthang@stanford.edu>
%
  % feed softmax vector of the previous timestep
  if params.feedInput
    x_t = [W_emb(:, wordId); softmax_h];
    
  % normal
  else
    x_t = W_emb(:, wordId);
  end
end