function [] = test(modelFile, beamSize)
  % modelFile = '/scr/nlp/deeplearning/lmthang/lstm/lstm.deen.50000.d1000.lr1.max5.d2.init0.1.noClip/modelRecent.mat';
  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));
  
  % trainLSTM('../data/id.1000/train.10k.sorted', '../data/id.1000/valid.100', '../data/id.1000/test.100', 'de', 'en', '../data/train.10k.de.vocab.1000', '../data/train.10k.en.vocab.1000', '../output', 0, 'logFreq', 10, 'numLayers', 2,'seed', 1, 'attnFunc', 0, 'isResume', 0)
  [savedData] = load(modelFile);
  params = savedData.params;
  model = savedData.model;
  
  printParams(2, params);
  
  % load test data
  [srcVocab] = params.vocab(params.tgtVocabSize+1:end);
  [tgtVocab] = params.vocab(1 : params.tgtVocabSize);
  testData  = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
  [candidates, scores] = lstmDecoder(model, testData.input, testData.inputMask, testData.srcMaxLen, params, beamSize);
  
  for i = 1:length(candidates)
    mask = find(testData.inputMask(i,1:testData.srcMaxLen));
    src = testData.input(i,mask(:));
    printSent(src, params.vocab, 'source: ');
    if isempty(candidates{i})
      fprintf(2, 'no translations.\n');
    else
      for j = 1 : length(candidates{i})
        printSent(candidates{i}{j}, params.vocab, 'candidate: ');
      end
    end
    fprintf(2, '=======================================\n');
  end
end
