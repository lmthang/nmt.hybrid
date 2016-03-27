function [] = testLSTM(modelFiles, beamSize, stackSize, batchSize, outputFile,varargin)
% Test a trained LSTM model by generating translations.
% Arguments:
%   modelFiles: single or multiple models to decode. Multiple models are
%     separated by commas.
%   beamSize: number of hypotheses kept at each time step.
%   stackSize: number of translations retrieved.
%   batchSize: number of sentences decoded simultaneously. We only ensure
%     accuracy of batchSize = 1 for now.
%   outputFile: output translation file.
%   varargin: other optional arguments.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
% Hieu Pham @ 2015, <hyhieu@cs.stanford.edu>

  addpath(genpath(sprintf('%s/..', pwd)));

  %% Argument Parser
  p = inputParser;
  % required
  addRequired(p,'modelFiles',@ischar);
  addRequired(p,'beamSize',@isnumeric);
  addRequired(p,'stackSize',@isnumeric);
  addRequired(p,'batchSize',@isnumeric);
  addRequired(p,'outputFile',@ischar);

  % optional
  addOptional(p,'gpuDevice', 0, @isnumeric); % choose the gpuDevice to use: 0 -- no GPU 
  addOptional(p,'align', 0, @isnumeric); % 1 -- output aignment from attention model
  addOptional(p,'assert', 0, @isnumeric); % 1 -- assert
  addOptional(p,'debug', 0, @isnumeric); % 1 -- debug
  addOptional(p,'minLenRatio', 0.5, @isnumeric); % decodeLen >= minLenRatio * srcMaxLen
  addOptional(p,'maxLenRatio', 1.5, @isnumeric); % decodeLen <= maxLenRatio * srcMaxLen
  addOptional(p,'testPrefix', '', @ischar); % to specify a different file for decoding
  addOptional(p,'hasTgt', 1, @isnumeric); % 0 -- no ref translations (groundtruth)
  addOptional(p,'continueId', 0, @isnumeric); % > 0: start decoding from this continueId (base 1) sent and append the results
  addOptional(p,'stopId', 0, @isnumeric); % > 0: stop decoding after finishing this stopId (base 1) sent
  addOptional(p,'computePpl', 0, @isnumeric); % 1 -- compute ppl of each participating model over the test set
    
  % force decoding: always feed the correct words (groundtruth)
  addOptional(p,'forceDecoder', 0, @isnumeric); 
  % useful for rescoring if we have many sentence pairs with the same source
  addOptional(p,'reuseEncoder', 0, @isnumeric); 
  % prefix decoding
  addOptional(p,'prefixFile', '', @ischar); % force the begining part to be equal to a prefix
  % print decoding scores, require stackSize = 1
  addOptional(p,'printScore', 0, @isnumeric);
  
  p.KeepUnmatched = true;
  parse(p,modelFiles,beamSize,stackSize,batchSize,outputFile,varargin{:})
  decodeParams = p.Results;
  assert(decodeParams.batchSize>0);
  
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
  
  %% load multiple models
  tokens = strsplit(decodeParams.modelFiles, ',');
  numModels = length(tokens);
  models = cell(numModels, 1);
  for mm=1:numModels
    [models{mm}] = loadDecodeModel(tokens{mm}, decodeParams);
    
    % make sure all models have the same vocab, and the number of layers
    if mm>1 
      for ii=1:models{mm}.params.srcVocabSize
        assert(strcmp(models{mm}.params.srcVocab{ii}, models{1}.params.srcVocab{ii}), '! model %d, mismatch src word %d: %s vs. %s\n', mm, ii, models{mm}.params.srcVocab{ii}, models{1}.params.srcVocab{ii});
      end
      for ii=1:models{mm}.params.tgtVocabSize
        assert(strcmp(models{mm}.params.tgtVocab{ii}, models{1}.params.tgtVocab{ii}), '! model %d, mismatch tgt word %d: %s vs. %s\n', mm, ii, models{mm}.params.tgtVocab{ii}, models{1}.params.tgtVocab{ii});
      end
      models{mm}.params = rmfield(models{mm}.params, {'srcVocab', 'tgtVocab'});
    end
  end
  
  params = models{1}.params;
  
  % reuse encoder (supposedly useful for batchSize = 1)
  if params.reuseEncoder
    assert(params.batchSize == 1);
  end
  
  
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
  % print score
  if params.printScore
    assert(params.stackSize == 1);
    params.scoreFid = fopen([params.outputFile '.score'], fileOpt);
  end
  
  printParams(2, params);
  
  % load test data  
  [srcSents, tgtSents, numSents]  = loadBiData(params, params.testPrefix, params.srcVocab, params.tgtVocab, -1, params.hasTgt);
  
  % compute perplexities
  if decodeParams.computePpl
    [testData] = prepareData(srcSents, tgtSents, 1, params);
    for mm=1:numModels
      evalValidTestSimple(models{mm}, testData, models{mm}.params);
    end
  end

  % force decode
  if decodeParams.forceDecoder
    fprintf(2, '# Force decoding\n');
    assert(params.batchSize == 1);
    assert(params.stackSize == 1);
  end
  
  % prefix decode
  if strcmp(params.prefixFile, '') == 0
    params.prefixDecoder = 1;
    assert(params.batchSize == 1);
    assert(params.stackSize == 1);
    
    fprintf(2, '# Prefix decoding\n');
    [prefixSents, ~] = loadMonoData(params.prefixFile, -1, params.baseIndex, params.tgtVocab, 'prefix');
    assert(length(prefixSents) == numSents);
  else
    params.prefixDecoder = 0;
  end
  

  
  
  %%%%%%%%%%%%
  %% decode %%
  %%%%%%%%%%%%
  numBatches = floor((numSents-1)/batchSize) + 1;
  
  fprintf(2, '# Decoding %d sents, %s\n', numSents, datestr(now));
  fprintf(params.logId, '# Decoding %d sents, %s\n', numSents, datestr(now));
  startTime = clock;
  otherInfo = [];
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
    [decodeData] = prepareData(srcSents(startId:endId), tgtSents(startId:endId), 1, params);
    decodeData.startId = startId;
    
    % prefix decoder
    if params.prefixDecoder || params.forceDecoder
      assert(startId == endId);
      if params.prefixDecoder
        decodeData.prefixSent = prefixSents{startId};
      else % like prefix decoder, prefixSent = tgtOutput
        decodeData.prefixSent = decodeData.tgtOutput(1, :);
      end
      decodeData.prefixLen = length(decodeData.prefixSent);
    else
      decodeData.prefixSent = [];
    end
    
    % call lstmDecoder
    [candidates, candScores, alignInfo, otherInfo] = lstmDecoder(models, decodeData, params, otherInfo); 
    
    % print results
    printDecodeResults(decodeData, candidates, candScores, alignInfo, params, 1, decodeData.prefixSent);
    if params.printScore
      fprintf(params.scoreFid, '%f\n', candScores); 
    end
    
    if startId == params.stopId % done
      break;
    end
  end

  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '# Complete decoding %d sents, time %.0fs, %s\n', numSents, timeElapsed, datestr(now));
  fprintf(params.logId, '# Complete decoding %d sents, time %.0fs, %s\n', numSents, timeElapsed, datestr(now));
  
  fclose(params.fid);
  fclose(params.logId);
  if params.printScore
    fclose(params.scoreFid);
  end
end

function [perp] = evalValidTestSimple(model, testData, params)
  fprintf(2, '  evaluating ...');
%   perp = 0;
  [testCosts] = evalCost(model, testData, params); % run on the test data
  testCounts = initCosts();
  testCounts = updateCounts(testCounts, testData);
  testCosts = scaleCosts(testCosts, testCounts);
  perp = exp(testCosts.word);
  fprintf(2, '  perp=%f\n', perp);
end
  

%     % convert local paths to absolute paths
%     fieldNames = fields(models{mm}.params);
%     for ii=1:length(fieldNames)
%       field = fieldNames{ii};
%       if ischar(models{mm}.params.(field))
%         if strfind(models{mm}.params.(field), '~lmthang/') ==1
%           models{mm}.params.(field) = strrep(models{mm}.params.(field), '~lmthang/', '/afs/ir/users/l/m/lmthang/');
%         end
%         if strfind(models{mm}.params.(field), '~lmthang/') ==1
%           models{mm}.params.(field) = strrep(models{mm}.params.(field), '~lmthang/', '/afs/cs.stanford.edu/u/lmthang/');
%         end
%         if strfind(models{mm}.params.(field), '~lmthang/') ==1
%           models{mm}.params.(field) = strrep(models{mm}.params.(field), '~lmthang/', '/home/lmthang/');
%         end    
%       end
%     end

