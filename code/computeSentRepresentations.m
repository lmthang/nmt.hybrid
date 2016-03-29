function [] = computeSentRepresentations(modelFile, outputFile,varargin)
% Test a trained LSTM model by generating translations.
% Arguments:
%   modelFile: single or multiple models to decode. Multiple models are
%     separated by commas.
%   batchSize: number of sentences decoded simultaneously. We only ensure
%     accuracy of batchSize = 1 for now.
%   outputFile: output translation file.
%   varargin: other optional arguments.
%
% Thang Luong @ 2016, <lmthang@stanford.edu>

  addpath(genpath(sprintf('%s/..', pwd)));

  %% Argument Parser
  p = inputParser;
  % required
  addRequired(p,'modelFile',@ischar);
  addRequired(p,'outputFile',@ischar);

  % optional
  addOptional(p,'gpuDevice', 0, @isnumeric); % choose the gpuDevice to use: 0 -- no GPU 
  addOptional(p,'align', 0, @isnumeric); % 1 -- output aignment from attention model
  addOptional(p,'assert', 0, @isnumeric); % 1 -- assert
  addOptional(p,'debug', 0, @isnumeric); % 1 -- debug
  addOptional(p,'testPrefix', '', @ischar); % to specify a different file for decoding
  addOptional(p,'continueId', 0, @isnumeric); % > 0: start decoding from this continueId (base 1) sent and append the results
    
  % useful for rescoring if we have many sentence pairs with the same source
  addOptional(p,'reuseEncoder', 0, @isnumeric);
  
  p.KeepUnmatched = true;
  parse(p,modelFile,outputFile,varargin{:})
  decodeParams = p.Results;
  
  % GPU settings
  decodeParams.isGPU = 0;
  if decodeParams.gpuDevice
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf(2, '# %d GPUs exist. So, we will use GPUs.\n', n);
      decodeParams.isGPU = 1;
      gpuDevice(decodeParams.gpuDevice)
      decodeParams.dataType = 'single';
    else
      decodeParams.dataType = 'double';
    end
  else
    decodeParams.dataType = 'double';
  end
  printParams(2, decodeParams);
  
  %% load model
  [model] = loadDecodeModel(decodeParams.modelFile, decodeParams);
  params = model.params;
  
  if params.continueId > 0 % appending
    fileOpt = 'a';
  else
    fileOpt = 'w';
  end
  
  params.fid = fopen(params.outputFile, fileOpt);
  params.logId = fopen([outputFile '.log'], fileOpt); 
  % align
  if params.align
    params.alignId = fopen([params.outputFile '.align'], fileOpt);
  end
  
  printParams(2, params);
  
  % default params
  params.hasTgt = 1;
  batchSize = 1;
  params.prefixDecoder = 0;
  
  % load test data  
  [srcSents, tgtSents, numSents]  = loadBiData(params, params.testPrefix, params.srcVocab, params.tgtVocab, -1, params.hasTgt);
  
  %%%%%%%%%%%%
  %% encode %%
  %%%%%%%%%%%%
  numBatches = floor((numSents-1)/batchSize) + 1;
  
  fprintf(2, '# Encoding %d sents, %s\n', numSents, datestr(now));
  fprintf(params.logId, '# Encoding %d sents, %s\n', numSents, datestr(now));
  startTime = clock;
  
  totalPredict = 0;
  encoderInfo = [];
  for batchId = 1 : numBatches
    % prepare batch data
    startId = (batchId-1)*batchSize+1;
    endId = batchId*batchSize;  
    if endId > numSents
      endId = numSents;
    end
    
    % continue training
    if params.continueId > startId
      continue;
    end    
    
    % prepare data
    [data] = prepareData(srcSents(startId:endId), tgtSents(startId:endId), 1, params);
    data.startId = startId;
    
    % encoding
    [prevStates, modelData, ~] = runEncoder(model, data, params, encoderInfo);
    if params.reuseEncoder
      encoderInfo.prevStates = prevStates;
      encoderInfo.modelData = modelData;
      encoderInfo.srcInput = data.srcInput;
    end
    
    for ii=1:(endId-startId+1)
      fprintf(params.fid, '%f ', prevStates{1}{end}.h_t(:, ii));
      fprintf(params.fid, '\n');
    end
    
    if mod(batchId, 100) == 0
      fprintf(2, '# Batch %d\n', batchId);
    end
  end

  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '# Complete encoding %d sents, num predict %d, ppl %g, time %.0fs, %s\n', numSents, totalPredict, ...
    timeElapsed, datestr(now));
  fprintf(params.logId, '# Complete encoding %d sents, num predict %d, ppl %g, time %.0fs, %s\n', numSents, totalPredict, ...
    timeElapsed, datestr(now));
  
  fclose(params.fid);
  fclose(params.logId);
end
