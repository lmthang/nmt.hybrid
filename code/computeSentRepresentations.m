function [] = computeSentRepresentations(modelFile, inFile, outputFile, varargin)
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
  addRequired(p,'inFile',@ischar);
  addRequired(p,'outputFile',@ischar);

  % optional
  addOptional(p,'gpuDevice', 0, @isnumeric); % choose the gpuDevice to use: 0 -- no GPU 
  addOptional(p,'align', 0, @isnumeric); % 1 -- output aignment from attention model
  addOptional(p,'assert', 0, @isnumeric); % 1 -- assert
  addOptional(p,'debug', 0, @isnumeric); % 1 -- debug
  addOptional(p,'testPrefix', '', @ischar); % to specify a different file for decoding
  addOptional(p,'continueId', 0, @isnumeric); % > 0: start decoding from this continueId (base 1) sent and append the results
  
  % opt 0: word-based models
  % opt 1: char-based models
  % opt 2: hybrid, require wordFile
  % opt 3: embedding lookup, require 'wordFile', ignore 'inFile'. For this
  % option, we will print word then embeddings
  
  addOptional(p,'opt', 0, @isnumeric); % 1 -- take src char model
  
  % for char opt 1 & 2, mix word embeddings and character-based embeddings.
  addOptional(p,'wordFile', '', @ischar); 
  
  % useful for rescoring if we have many sentence pairs with the same source
  addOptional(p,'reuseEncoder', 0, @isnumeric);
  
  p.KeepUnmatched = true;
  parse(p,modelFile,inFile,outputFile,varargin{:})
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
  model.params.attnFunc = 0;
  params = model.params;
  
  % look up W_emb_src for frequent words
  useWordEmbs = 0;
  if params.opt == 2 || params.opt == 3
  	assert(strcmp(params.wordFile, '') == 0);
    
    fprintf(2, '# Loading wordFile %s\n', params.wordFile);
    [words, ~] = loadVocab(params.wordFile);   
    
    if params.opt == 3 % word-embedding lookup
      unk = '<unk>';
      assert(ismember(unk, params.srcVocab) == 1);
      if ismember(unk, words) == 0
        words{end+1} = unk;
        fprintf(2, '  appending %s at the end\n', unk);
      end
      [word_flags, word_positions] = ismember(words, params.srcVocab);
    else % hybrid
      assert(params.charOpt > 0); 
      [word_flags, word_positions] = ismember(words, params.srcVocab(1:params.srcCharShortList));
    end
    
    word_embs = zeroMatrix([params.lstmSize, length(word_flags)], params.isGPU, params.dataType);
    word_embs(:, word_flags) = model.W_emb_src(:, word_positions(word_flags));
    fprintf(2, '  num overlapped words %d\n', size(word_embs, 2));
    
    useWordEmbs = 1;
  end
    
  if params.opt == 1 || params.opt == 3 % char or hybrid
    % model
    charModel.W_src = model.W_src_char;
    charModel.W_emb_src = model.W_emb_src_char;
    
    % params
    params.srcVocab = params.srcCharVocab;
    params.srcVocabSize = params.srcCharVocabSize;
    params.numLayers = params.charNumLayers;
    charModel.params = params;
    
    model = charModel;
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
  
  printParams(2, params);
  
  % default params
  params.hasTgt = 1;
  batchSize = 1;
  params.prefixDecoder = 0;
  
  % load test data  
  if params.opt == 1 || params.opt == 2
    [sents, numSents] = loadMonoData(params.inFile, -1, 0, params.srcVocab, 'src');
    if useWordEmbs
      assert(numSents == length(word_flags));
    end
  else
    numSents = length(words);
  end
  
  
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
    
    if params.opt == 3 % special opt, read above, word embedding lookup only
      assert(useWordEmbs && batchSize == 1);
      if word_flags(startId) % look up frequent words for hybrid models
        fprintf(params.fid, '%s', words{startId});
        fprintf(params.fid, ' %f', word_embs(:, startId));
        fprintf(params.fid, '\n');
      end
    else
      if useWordEmbs && batchSize == 1 && word_flags(startId) % look up frequent words for hybrid models
        assert(params.opt == 2);
        fprintf(params.fid, '%f ', word_embs(:, startId));
        fprintf(params.fid, '\n');
      else
        % prepare data
        if params.opt == 1 || params.opt == 2 % char
          [data] = prepareMonoData(sents(startId:endId), params.srcCharSos, params.srcCharEos, -1, 1);
        else
          [data] = prepareMonoData(sents(startId:endId), params.srcSos, -1, -1, 1);
        end

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
      end
    end
    
    if mod(batchId, 100) == 0
      fprintf(2, '# Batch %d\n', batchId);
    end
  end

  endTime = clock;
  timeElapsed = etime(endTime, startTime);
  fprintf(2, '# Complete encoding %d sents, num predict %d, time %.0fs, %s\n', numSents, totalPredict, ...
    timeElapsed, datestr(now));
  fprintf(params.logId, '# Complete encoding %d sents, num predict %d, time %.0fs, %s\n', numSents, totalPredict, ...
    timeElapsed, datestr(now));
  
  fclose(params.fid);
  fclose(params.logId);
end
