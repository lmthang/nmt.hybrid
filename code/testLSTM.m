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
  addOptional(p,'gpuDevice', 1, @isnumeric); % choose the gpuDevice to use. 
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
  
  for batchId = 1 : numBatches
    % prepare batch data
    startId = (batchId-1)*batchSize+1;
    endId = batchId*batchSize;
    if endId > numSents
      endId = numSents;
    end
    [decodeData] = prepareData(srcSents(startId:endId), tgtSents(startId:endId), params);
    decodeData.sentIndices = startId:endId;
    
    % call lstmDecoder
    [candidates, scores] = lstmDecoder(model, decodeData, params, beamSize, stackSize); 
    
    % output translations
    for ii = 1:length(candidates)
      [maxScore, bestId] = max(scores{ii});
      translation = candidates{ii}{bestId}(1:end-1); % remove <t_eos>
      printSent(params.fid, translation, params.vocab, ''); 
      
      % log
      printSrc(params.logId, decodeData, ii, params);
      printRef(params.logId, decodeData, ii, params);
      printSent(params.logId, translation, params.vocab, '');
      
      % print debug info
      printSrc(2, decodeData, ii, params);
      printRef(2, decodeData, ii, params);
      printSent(2, candidates{ii}{bestId}, params.vocab, ['best ' num2str(maxScore) ': ']);
      printTranslations(candidates{ii}, scores{ii}, params);
    end  
  end

  fclose(params.fid);
  fclose(params.logId);
end


function printSrc(fid, testData, ii, params)
  mask = testData.inputMask(ii,1:testData.srcMaxLen);
  src = testData.input(ii,mask);
  printSent(fid, src, params.vocab, ['# source ' num2str(ii) ': ']);
end

function printRef(fid, testData, ii, params)
  mask = testData.inputMask(ii, testData.srcMaxLen:end);
  ref = testData.tgtOutput(ii,mask);
  printSent(fid, ref, params.vocab, ['# ref ' num2str(ii) ': ']);
end

function printTranslations(candidates, scores, params)
  for jj = 1 : length(candidates)
    printSent(2, candidates{jj}, params.vocab, ['cand ' num2str(jj) ', ' num2str(scores(jj)) ': ']);
  end
end



