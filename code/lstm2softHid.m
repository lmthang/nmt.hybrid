function [softmax_h, attn_h_concat, alignWeights, alignScores, attnInput] = lstm2softHid(h_t, params, model, varargin)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if params.softmaxDim>0 % f(W_h * h_t)
    softmax_h = params.nonlinear_f(model.W_h*h_t);
  else  
    if params.attnFunc>0 % attention mechanism
      srcAlignStates = varargin{1};
      mask = varargin{2};
      curBatchSize = varargin{3};
      srcLens = varargin{4};
      [softmax_h, attn_h_concat, alignWeights, alignScores, attnInput] = attnForward(h_t, model, srcAlignStates, mask, params, curBatchSize, srcLens);
    else % normal
      softmax_h = h_t;
    end
  end
end