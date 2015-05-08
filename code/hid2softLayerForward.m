function [softmax_h, h2sInfo] = hid2softLayerForward(h_t, params, model, trainData, curMask, tgtPos)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
  if params.posModel>=1 && mod(tgtPos, 2)==1 % positions
    isPredictPos = 1;
  else % words
    isPredictPos = 0;
  end
  
  h2sInfo = [];
  
  
  % softmax compression: f(W_h * h_t)
  if params.softmaxDim>0
    softmax_h = hiddenLayerForward(model.W_h, h_t, params.nonlinear_f);
    
  % positional model 3: f(W_h * [srcPosVecs; h_t])
  elseif params.posModel==3 && isPredictPos==0 
    predWords = trainData.tgtOutput(:, tgtPos-1)'; % Here we look at the previous time steps for positions
    predWords = predWords - params.startPosId + 1;
    [srcHidVecs, h2sInfo.linearIndices] = buildSrcPosVecs(tgtPos, params, trainData, predWords, curMask);
    h2sInfo.input = [srcHidVecs; h_t];
    softmax_h = hiddenLayerForward(model.W_h, h2sInfo.input, params.nonlinear_f);
  % attention mechanism
  elseif params.attnFunc>0
    if params.attnRelativePos % relative
      [srcHidVecs, h2sInfo.startAttnId, h2sInfo.endAttnId, h2sInfo.startHidId, h2sInfo.endHidId] = buildSrcHidVecs(trainData.srcHidVecs, trainData.srcMaxLen, tgtPos, params);
    else
      srcHidVecs = trainData.absSrcHidVecs;
    end
    
    % softmax_h = f(W_ah*[attn_t; tgt_h_t])
    [attnVecs, h2sInfo.alignWeights] = attnLayerForward(model.W_a, h_t, srcHidVecs, curMask.mask);
    h2sInfo.attn_h_concat = [attnVecs; h_t];
    [softmax_h] = hiddenLayerForward(model.W_ah, h2sInfo.attn_h_concat, params.nonlinear_f);

  % no transition
  else
    softmax_h = h_t;
  end
end
    