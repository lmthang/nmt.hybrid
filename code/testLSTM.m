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

  %% Argument Parser
  p = inputParser;
  % required
  addRequired(p,'modelFile',@ischar);
  addRequired(p,'beamSize',@isnumeric);
  addRequired(p,'stackSize',@isnumeric);
  addRequired(p,'batchSize',@isnumeric);
  addRequired(p,'outputFile',@ischar);

  % optional
  addOptional(p,'unkPenalty', 1, @isnumeric); % in log domain unkPenalty=0.5 ~ scale prob unk by 1.6
  addOptional(p,'unkId', 1, @isnumeric); % id of unk word
  addOptional(p,'gpuDevice', 1, @isnumeric); % choose the gpuDevice to use. 
  addOptional(p,'lengthReward', 0.5, @isnumeric); % in log domain, promote longer sentences.
  p.KeepUnmatched = true;
  parse(p,modelFile,beamSize,stackSize,batchSize,outputFile,varargin{:})
  decodeParams = p.Results;

  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));
  printParams(2, decodeParams);

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
  
  [savedData] = load(decodeParams.modelFile);
  params = savedData.params;
  params.posModel=0;
  model = savedData.model;
  params.unkPenalty = decodeParams.unkPenalty;
  params.unkId = decodeParams.unkId;
  params.lengthReward = decodeParams.lengthReward;
  assert(strcmp(params.vocab{params.unkId}, '<unk>')==1);
  model
  
  % check GPUs
  params.isGPU = decodeParams.isGPU;
  params.dataType = decodeParams.dataType;
  printParams(2, params);
  
  % load test data
  [srcVocab] = params.vocab(params.tgtVocabSize+1:end);
  [tgtVocab] = params.vocab(1 : params.tgtVocabSize);
  [srcSents, tgtSents, numSents]  = loadBiData(params, params.testPrefix, srcVocab, tgtVocab);
  if decodeParams.batchSize==-1 % decode all sents at once if no batchSize is specified
    decodeParams.batchSize = numSents;
  end
  
  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%
  params.fid = fopen(decodeParams.outputFile, 'w');
  params.logId = fopen([outputFile '.log'], 'w');
  numBatches = floor((numSents-1)/batchSize) + 1;
  
  fprintf(2, '# Decoding %d sents, %s\n', numSents, datestr(now));
  fprintf(params.logId, '# Decoding %d sents, %s\n', numSents, datestr(now));
  startTime = clock;
  for batchId = 1 : numBatches
    % prepare batch data
    startId = (batchId-1)*batchSize+1;
    endId = batchId*batchSize;
    
%     if endId>10
%       break;
%     end
    
    if endId > numSents
      endId = numSents;
    end
    [decodeData] = prepareData(srcSents(startId:endId), tgtSents(startId:endId), params);
    decodeData.sentIndices = startId:endId;
    
    % call lstmDecoder
    [candidates, candScores] = lstmDecoder(model, decodeData, params, beamSize, stackSize); 
    
    % output translations
    [maxScores, bestIndices] = max(candScores); % stackSize * batchSize
    curBatchSize = endId-startId+1;
    for ii = 1:curBatchSize
      bestId = bestIndices(ii);
      translation = candidates{ii}{bestId}; 
      
      assert(isempty(find(translation>params.tgtVocabSize, 1)));
      printSent(params.fid, translation(1:end-1), params.vocab, ''); % remove <t_eos>
      
      % log
      printSrc(params.logId, decodeData, ii, params, startId+ii-1);
      printRef(params.logId, decodeData, ii, params, startId+ii-1);
      printSent(params.logId, translation, params.vocab, ['  tgt ' num2str(startId+ii-1) ': ']);
      fprintf(params.logId, '  score %g\n', maxScores(ii));
      
      % debug
      if ii==curBatchSize
        printSrc(2, decodeData, ii, params, startId+ii-1);
        printRef(2, decodeData, ii, params, startId+ii-1);
        printSent(2, translation, params.vocab, ['  tgt ' num2str(startId+ii-1) ': ']);
        fprintf(2, '  score %g\n', maxScores(ii));
        %printTranslations(candidates{ii}, candScores(ii, :), params);
      end
    end  
    
  end

  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '# Complete decoding %d sents, time %.0fs, %s\n', numSents, timeElapsed, datestr(now));
  fprintf(params.logId, '# Complete decoding %d sents, time %.0fs, %s\n', numSents, timeElapsed, datestr(now));
  
  fclose(params.fid);
  fclose(params.logId);
end


function printSrc(fid, testData, ii, params, sentId)
  mask = testData.inputMask(ii,1:testData.srcMaxLen);
  src = testData.input(ii,mask);
  printSent(fid, src, params.vocab, ['# src ' num2str(sentId) ': ']);
end

function printRef(fid, testData, ii, params, sentId)
  mask = testData.inputMask(ii, testData.srcMaxLen:end);
  ref = testData.tgtOutput(ii,mask);
  printSent(fid, ref, params.vocab, ['  ref ' num2str(sentId) ': ']);
end

function printTranslations(candidates, scores, params)
  for jj = 1 : length(candidates)
    assert(isempty(find(candidates{jj}>params.tgtVocabSize, 1)));
    printSent(2, candidates{jj}, params.vocab, ['cand ' num2str(jj) ', ' num2str(scores(jj)) ': ']);
  end
end



