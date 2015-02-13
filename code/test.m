function [] = test(modelFile, beamSize, stackSize, batchSize, outputFile,varargin)
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
  
  % check GPUs
  params.isGPU = decodeParams.isGPU;
  params.dataType = decodeParams.dataType;
  printParams(2, params);
  
  % load test data
  [srcVocab] = params.vocab(params.tgtVocabSize+1:end);
  [tgtVocab] = params.vocab(1 : params.tgtVocabSize);
  testData  = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  model

  if decodeParams.batchSize==-1
    decodeParams.batchSize = size(testData.input, 1);
  end
  params.fid = fopen(decodeParams.outputFile, 'w');
  decode(model, testData, params, decodeParams.beamSize, decodeParams.stackSize, decodeParams.batchSize);
  

  fclose(params.fid);
end

function decode(model, data, params, beamSize, stackSize, batchSize) %[allCandidates, allScores] = 
  numSents = size(data.input, 1);
  numBatches = floor((numSents-1)/batchSize) + 1;
  %allCandidates = cell(numSents, 1); 
  %allScores = cell(numSents, 1);
  
  decodeData.srcMaxLen = data.srcMaxLen;
  decodeData.tgtMaxLen = data.tgtMaxLen;
  for batchId = 1 : numBatches
    startId = (batchId-1)*batchSize+1;
    endId = batchId*batchSize;
    if endId > numSents
      endId = numSents;
    end
    
    decodeData.input = data.input(startId:endId, :);
    decodeData.inputMask = data.inputMask(startId:endId, :);
    decodeData.tgtOutput = data.tgtOutput(startId:endId, :);
    decodeData.srcLens = data.srcLens(startId:endId);
    decodeData.sentIndices = startId:endId;
    
    
    [candidates, scores] = lstmDecoder(model, decodeData, params, beamSize, stackSize); 
    
    %allCandidates(startId:endId) = candidates;
    %allScores(startId:endId) = scores;
    for ii = 1:length(candidates)
      [maxScore, bestId] = max(scores{ii});
      printSent(params.fid, candidates{ii}{bestId}(1:end-1), params.vocab, ''); % remove <t_eos>
      printSrc(decodeData, ii, params);
      printRef(decodeData, ii, params);
      printSent(2, candidates{ii}{bestId}, params.vocab, ['best ' num2str(maxScore) ': ']);
      printTranslations(candidates{ii}, scores{ii}, params);
    end  
  end
end

function printSrc(testData, ii, params)
  mask = testData.inputMask(ii,1:testData.srcMaxLen);
  src = testData.input(ii,mask);
  printSent(2, src, params.vocab, ['# source ' num2str(ii) ': ']);
end

function printRef(testData, ii, params)
  mask = testData.inputMask(ii, testData.srcMaxLen:end);
  ref = testData.tgtOutput(ii,mask);
  printSent(2, ref, params.vocab, ['# ref ' num2str(ii) ': ']);
end

function printTranslations(candidates, scores, params)
  for jj = 1 : length(candidates)
    printSent(2, candidates{jj}, params.vocab, ['cand ' num2str(jj) ', ' num2str(scores(jj)) ': ']);
  end
end



