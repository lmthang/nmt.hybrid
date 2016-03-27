function [prevStates, modelData, models] = runEncoder(models, data, params, encoderInfo)
% Compute encoder representations
% Thang Luong @ 2015, <lmthang@stanford.edu>

  % backward compatibility: not a cell, single model, put into a cell format
  if ~iscell(models)
    tmpModels = cell(1, 1);
    tmpModels{1} = models;
    tmpModels{1}.params = params;
    models = tmpModels;
  end
  
  batchSize = size(data.srcInput, 1);
  srcMaxLen = data.srcMaxLen;
  params.srcMaxLen = srcMaxLen;

  % fprintf(2, '# Encoding batch of %d sents, srcMaxLen=%d, %s\n', batchSize, srcMaxLen, datestr(now));
  % fprintf(params.logId, '# Encoding batch of %d sents, srcMaxLen=%d, %s\n', batchSize, srcMaxLen, datestr(now));

  %% multiple models
  numModels = length(models);
  modelData = cell(numModels, 1);
  zeroStates = cell(numModels, 1);
  for mm=1:numModels
    models{mm}.params.curBatchSize = batchSize;
    models{mm}.params.srcMaxLen = srcMaxLen;
    [models{mm}.params] = setAttnParams(models{mm}.params);

    [zeroStates{mm}] = createZeroState(models{mm}.params);  
  end
  
  % reuse encoder
  if ~isempty(encoderInfo) && params.reuseEncoder && isequal(data.srcInput, encoderInfo.srcInput)
    prevStates = encoderInfo.prevStates;
    modelData = encoderInfo.modelData;
  else
    prevStates = cell(numModels, 1);
    for mm=1:numModels
      encRnnFlags = struct('decode', 0, 'test', 1, 'attn', models{mm}.params.attnFunc, 'feedInput', 0);
      [encStates, modelData{mm}, ~] = rnnLayerForward(models{mm}.W_src, models{mm}.W_emb_src, zeroStates{mm}, data.srcInput, ...
        data.srcMask, models{mm}.params, encRnnFlags, data, models{mm});
      prevStates{mm} = encStates{end};

      % feed input
      if models{mm}.params.feedInput
        prevStates{mm}{end}.softmax_h = zeroMatrix([models{mm}.params.lstmSize, batchSize], params.isGPU, params.dataType);
      end
    end
  end
end