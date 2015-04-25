function [] = testLSTM(modelFile, beamSize, stackSize, batchSize, outputFile,varargin)
%%%
%
% Test a trained LSTM model by generating translations for the test data
% (stored under the params structure in the model file)
%   stackSize: the maximum number of translations we want to get.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>
%
%%%
  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));

  %% Argument Parser
  p = inputParser;
  % required
  addRequired(p,'modelFile',@ischar);
  addRequired(p,'beamSize',@isnumeric);
  addRequired(p,'stackSize',@isnumeric);
  addRequired(p,'batchSize',@isnumeric);
  addRequired(p,'outputFile',@ischar);

  % optional
  addOptional(p,'gpuDevice', 1, @isnumeric); % choose the gpuDevice to use. 
  addOptional(p,'minLenRatio', 0.5, @isnumeric); % decodeLen >= minLenRatio * srcMaxLen
  addOptional(p,'maxLenRatio', 1.5, @isnumeric); % decodeLen <= maxLenRatio * srcMaxLen
  addOptional(p,'depParse', 0, @isnumeric); % 1: indicate that we are doing dependency parsing
  addOptional(p,'depRootId', -1, @isnumeric); % 1: indicate that we are doing dependency parsing
  addOptional(p,'testPrefix', '', @ischar); % to specify a different file for decoding

  p.KeepUnmatched = true;
  parse(p,modelFile,beamSize,stackSize,batchSize,outputFile,varargin{:})
  decodeParams = p.Results;
  if decodeParams.batchSize==-1 % decode sents one by one
    decodeParams.batchSize = 1;
  end
  
  % dependency parsing
  if decodeParams.depParse 
    fprintf(2, '## Dependency parsing\n');
    assert(decodeParams.batchSize==1);
  end
  
  % GPU settings
  decodeParams.isGPU = 0;
  if ismac==0
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
  
  % load model
  [savedData] = load(decodeParams.modelFile);
  params = savedData.params;  
  params.posModel=0;
  model = savedData.model;
  model
 
  % for backward compatibility
  if ~isfield(params, 'numClasses')
    params.numClasses = 0;
  end
  if ~isfield(params, 'dropout')
    params.dropout = 1;
  end
  
  % convert absolute paths to local paths
  fieldNames = fields(params);
  for ii=1:length(fieldNames)
    field = fieldNames{ii};
    if ischar(params.(field))
      if strfind(params.(field), '/afs/ir/users/l/m/lmthang') ==1
        params.(field) = strrep(params.(field), '/afs/ir/users/l/m/lmthang', '~');
      end
      if strfind(params.(field), '/afs/cs.stanford.edu/u/lmthang') ==1
        params.(field) = strrep(params.(field), '/afs/cs.stanford.edu/u/lmthang', '~');
      end
    end
  end
  
  if ~isfield(params, 'separateEmb')
    params.separateEmb = 0;
  end
  [params] = loadBiVocabs(params);
  params.vocab = [params.tgtVocab params.srcVocab];
  % copy fields
  fieldNames = fields(decodeParams);
  for ii=1:length(fieldNames)
    field = fieldNames{ii};
    if strcmp(field, 'testPrefix')==1 && strcmp(decodeParams.(field), '')==1 % skip empty testPrefix
      continue;
    elseif strcmp(field, 'testPrefix')==1
      fprintf(2, '# Decode a different test file %s\n', decodeParams.(field));
    end
    params.(field) = decodeParams.(field);
  end
  
  if ~isfield(params, 'softmaxDim')
    params.softmaxDim = 0;
  end
  params.fid = fopen(params.outputFile, 'w');
  params.logId = fopen([outputFile '.log'], 'w');
  printParams(2, params);
  
  % load test data
  [srcSents, tgtSents, numSents]  = loadBiData(params, params.testPrefix, params.srcVocab, params.tgtVocab);
  %[srcSents, tgtSents, numSents]  = loadBiData(params, params.trainPrefix, srcVocab, tgtVocab, 10);
  
  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%
  numBatches = floor((numSents-1)/batchSize) + 1;
  
  fprintf(2, '# Decoding %d sents, %s\n', numSents, datestr(now));
  fprintf(params.logId, '# Decoding %d sents, %s\n', numSents, datestr(now));
  startTime = clock;
  for batchId = 1 : numBatches
    % prepare batch data
    startId = (batchId-1)*batchSize+1;
    endId = batchId*batchSize;
    
    if endId > numSents
      endId = numSents;
    end
    [decodeData] = prepareData(srcSents(startId:endId), tgtSents(startId:endId), 1, params);
    decodeData.startId = startId;
    
    % call lstmDecoder
    [candidates, candScores] = lstmDecoder(model, decodeData, params); 
    
    % print results
    printDecodeResults(decodeData, candidates, candScores, params, 1);
  end

  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '# Complete decoding %d sents, time %.0fs, %s\n', numSents, timeElapsed, datestr(now));
  fprintf(params.logId, '# Complete decoding %d sents, time %.0fs, %s\n', numSents, timeElapsed, datestr(now));
  
  fclose(params.fid);
  fclose(params.logId);
end


%   addOptional(p,'option', 0, @isnumeric); % 0: normal, 1: depparse, 2: permutation
%   addOptional(p,'unkId', 1, @isnumeric); % id of unk word
%   addOptional(p,'accmLstm', 0, @isnumeric); % 1: accmulate h_t/c_t as we go over the src side.
%   addOptional(p,'unkPenalty', 0, @isnumeric); % in log domain unkPenalty=0.5 ~ scale prob unk by 1.6
%   addOptional(p,'lengthReward', 0, @isnumeric); % in log domain, promote longer sentences.
  
%   assert(strcmp(params.vocab{params.unkId}, '<unk>')==1);

%   if decodeParams.option==1 % depparse
%     decodeParams.minLenRatio = 1.5;
%     decodeParams.maxLenRatio = 2.5;
%   else if decodeParams.option==2 % permutation
%     decodeParams.minLenRatio = 1;
%     decodeParams.maxLenRatio = 1;
%   end


