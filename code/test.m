function [] = test(modelFile, beamSize, stackSize, batchSize, outputFile)
  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));
  
  [savedData] = load(modelFile);
  params = savedData.params;
  model = savedData.model;
  
  printParams(2, params);
  
  % load test data
  [srcVocab] = params.vocab(params.tgtVocabSize+1:end);
  [tgtVocab] = params.vocab(1 : params.tgtVocabSize);
  testData  = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
  [allCandidates, allScores] = decode(model, testData, params, beamSize, stackSize, batchSize);
  
  fid = fopen(outputFile, 'w');
  for ii = 1:length(allCandidates)
    [maxScore, bestId] = max(allScores{ii});
    printSent(fid, allCandidates{ii}{bestId}(1:end-1), params.vocab, ''); % remove <t_eos>
    printSrc(testData, ii, params);
    printSent(2, allCandidates{ii}{bestId}, params.vocab, ['best ' num2str(maxScore) ': ']);
    printTranslations(allCandidates{ii}, allScores{ii}, params);
  end
  fclose(fid);
end

function printTranslations(candidates, scores, params)
  for jj = 1 : length(candidates)
    printSent(2, candidates{jj}, params.vocab, ['cand ' num2str(jj) ', ' num2str(scores(jj)) ': ']);
  end
end

function printSrc(testData, ii, params)
  mask = find(testData.inputMask(ii,1:testData.srcMaxLen));
  src = testData.input(ii,mask(:));
  printSent(2, src, params.vocab, ['# source ' num2str(ii) ': ']);
end

function [allCandidates, allScores] = decode(model, data, params, beamSize, stackSize, batchSize)
  numSents = size(data.input, 1);
  numBatches = floor((numSents-1)/batchSize) + 1;
  allCandidates = cell(numSents, 1); 
  allScores = cell(numSents, 1);
  
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
    allCandidates(startId:endId) = candidates;
    allScores(startId:endId) = scores;
  end
end
