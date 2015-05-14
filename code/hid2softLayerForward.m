function [softmax_h, h2sInfo] = hid2softLayerForward(h_t, params, model, trainData, curMask, tgtPos, isTest)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
  h2sInfo = [];
  
  if params.softmaxDim || params.attnFunc || (params.posModel==3 && mod(tgtPos, 2)==0)
    if params.softmaxDim % softmax compression: f(W_h * h_t)
      h2sInfo.input = h_t;
    elseif params.posModel==3 && mod(tgtPos, 2)==0 % positional model 3: f(W_h * [srcPosVecs; h_t])
      if isTest==0
        positions = trainData.tgtOutput(:, tgtPos-1)'; % Here we look at the previous time steps for positions
      else
        positions = trainData.positions;
      end
      
      [srcHidVecs, h2sInfo.linearIndices] = buildSrcPosVecs(tgtPos, params, trainData, positions, curMask);
      h2sInfo.input = [srcHidVecs; h_t];
    elseif params.attnFunc % attention mechanism: f(W_h*[attn_t; tgt_h_t])
      if params.attnRelativePos % relative
        [srcHidVecs, h2sInfo.startAttnId, h2sInfo.endAttnId, h2sInfo.startHidId, h2sInfo.endHidId] = buildSrcHidVecs(...
          trainData.srcHidVecs, trainData.srcMaxLen, tgtPos, params);
      else
        srcHidVecs = trainData.absSrcHidVecs;
      end

      h2sInfo.h_t = h_t;
      [attnVecs, h2sInfo.alignWeights] = attnLayerForward(model.W_a, h_t, srcHidVecs, curMask.mask);
      h2sInfo.input = [attnVecs; h_t];
    end
    
    softmax_h = hiddenLayerForward(model.W_h, h2sInfo.input, params.nonlinear_f);
    h2sInfo.softmax_h = softmax_h;
  else % no intermediate layer
    softmax_h = h_t;
  end
end
    