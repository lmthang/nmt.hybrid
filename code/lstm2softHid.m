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
      trainData = varargin{1};
      [softmax_h, attn_h_concat, alignWeights, alignScores, attnInput] = attnForward(h_t, model, params, trainData);
    else % normal
      softmax_h = h_t;
    end
  end
end