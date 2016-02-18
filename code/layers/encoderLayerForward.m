function [encStates, lastEncState, encRnnFlags, trainData, srcCharData] = encoderLayerForward(model, initState, trainData, params, isTest)
  % char
  srcCharData = [];
  if params.charOpt
    % src
    if params.charSrcRep
      [srcCharData] = srcCharLayerForward(model.W_src_char, model.W_emb_src_char, trainData.srcInput, trainData.srcMask, params.srcCharMap, ...
        params.srcVocabSize, params, isTest);
    else
      trainData.srcInput(trainData.srcInput > params.srcCharShortList) = params.srcUnk;
    end
  end

  % rnn
  encRnnFlags = struct('decode', 0, 'test', isTest, 'attn', params.attnFunc, 'feedInput', 0, 'charSrcRep', params.charSrcRep, ...
    'charTgtGen', params.charTgtGen, 'initEmb', []);
  [encStates, trainData, ~] = rnnLayerForward(model.W_src, model.W_emb_src, initState, trainData.srcInput, trainData.srcMask, ...
    params, encRnnFlags, trainData, model, srcCharData);
  lastEncState = encStates{end};

  % feed input
  if params.feedInput
    lastEncState{end}.softmax_h = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  end
end