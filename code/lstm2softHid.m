function [softmax_h, input, attn_h_concat, alignWeights] = lstm2softHid(h_t, params, model, varargin)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if params.softmaxDim>0 % compression: f(W_h * h_t)
    input = h_t;
    softmax_h = params.nonlinear_f(model.W_h*h_t);
  elseif params.attnFunc>0 % attention mechanism
    srcHidVecs = varargin{1};
    curMask = varargin{2};

    [softmax_h, attn_h_concat, alignWeights] = attnForward(h_t, model, params, srcHidVecs, curMask);
    input = h_t;
  elseif params.posModel==3 % positional model: f(W_h * [srcPosVecs; h_t])
    isPredictPos = varargin{1};
    if isPredictPos==0
      srcPosVecs = varargin{2};
      input = [srcPosVecs; h_t];
      softmax_h = params.nonlinear_f(model.W_h*input);
    else
      input = [];
      softmax_h = h_t;
    end
  else % normal
    input = [];
    softmax_h = h_t;
  end
end