function [softmax_h, hid2softData] = hid2softForward(h_t, params, model, trainData, curMask, curInfo)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%  
  
  hid2softData.input = h_t;
  hid2softData.srcHidVecs = [];
  if params.attnFunc>0 && params.attnFeedSoftmax % attention mechanism
    if params.attnRelativePos % relative
      [hid2softData.startAttnId, hid2softData.endAttnId, hid2softData.startHidId, hid2softData.endHidId] = buildSrcHidVecs(trainData.srcMaxLen, curInfo.tgtPos, params);
      hid2softData.srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
      hid2softData.srcHidVecs(:, :, hid2softData.startHidId:hid2softData.endHidId) = trainData.srcHidVecs(:, :, hid2softData.startAttnId:hid2softData.endAttnId);
    else
      hid2softData.srcHidVecs = trainData.absSrcHidVecs;
    end
    
    % softmax_h = f(W_ah*[attn_t; tgt_h_t])
    [attnVecs, hid2softData.alignWeights] = attnLayerForward(model.W_a, h_t, hid2softData.srcHidVecs, curMask.mask);
    hid2softData.attn_h_concat = [attnVecs; h_t];
    [softmax_h] = hiddenLayerForward(model.W_ah, hid2softData.attn_h_concat, params.nonlinear_f);
  elseif params.softmaxDim>0 || (params.posModel==3 && curInfo.isPredictPos==0) % compression: f(W_h * h_t) or positional model: f(W_h * [srcPosVecs; h_t])
    if params.posModel==3 % refer to src info
      [srcHidVecs, hid2softData.linearIndices] = buildSrcPosVecs(curInfo.tt, params, trainData, curInfo.predWords, curMask);
      hid2softData.input = [srcHidVecs; h_t];
    end
    softmax_h = hiddenLayerForward(model.W_h, hid2softData.input, params.nonlinear_f);
  else
    softmax_h = h_t;
  end
end
    