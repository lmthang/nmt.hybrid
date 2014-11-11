function trainLSTM(trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,baseIndex,varargin)
%%%
%
% Train Long-Short Term Memory (LSTM).
% Options:
%   srcLang, srcVocabFile: leave empty to train monolingual models.
%   baseIndex: of training data. Required to convert them to 1-indexed.
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%
  
  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));
  
  %% Argument Parser %%
  p = inputParser;
  % required
  addRequired(p,'trainPrefix',@ischar);
  addRequired(p,'validPrefix',@ischar);
  addRequired(p,'testPrefix',@ischar);
  addRequired(p,'srcLang',@ischar);
  addRequired(p,'tgtLang',@ischar);
  addRequired(p,'srcVocabFile',@ischar);
  addRequired(p,'tgtVocabFile',@ischar);
  addRequired(p,'outDir',@ischar);
  addRequired(p,'baseIndex',@isnumeric);
  % optional
  addOptional(p,'lstmSize', 100, @isnumeric);
  addOptional(p,'learningRate', 1.0, @isnumeric);
  addOptional(p,'maxGradNorm', 5.0, @isnumeric);
  addOptional(p,'initRange', 0.1, @isnumeric);
  addOptional(p,'batchSize', 128, @isnumeric);
  addOptional(p,'numEpoches', 5, @isnumeric); % num epoches
  addOptional(p,'epochFraction', 0.5, @isnumeric);
  addOptional(p,'finetuneEpoch', 3, @isnumeric); % epoch > finetuneEpoch, start halving learning rate every epochFraction of an epoch, e.g., every 0.5 epoch
  addOptional(p,'logFreq', 10, @isnumeric); % how frequent (number of batches) we want to log stuffs
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need to specify other input arguments as toy data is automatically generated.
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'isBi', 1, @isnumeric); % isBi=0: mono model, isBi=1: bi (encoder-decoder) model
  addOptional(p,'isClip', 0, @isnumeric); % isClip=1: clip forward 50, clip backward 1000
  addOptional(p,'globalOpt', 0, @isnumeric); % globalOpt=0: no global model, 1: avg global model, 2: feedforward global model.
  
  p.KeepUnmatched = true;
  parse(p,trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,baseIndex,varargin{:})
  
  params = p.Results;
  params.chunkSize = params.batchSize*100;
  
  % clip
  params.clipForward = 50; % clip c_t, h_t
  params.clipBackward = 1000; % clip dc, dh
  
  % act functions for gate
  params.nonlinear_gate_f = @sigmoid;
  params.nonlinear_gate_f_prime = @sigmoidPrime;
  
  % act functions for others
  params.nonlinear_f = @tanh;
  params.nonlinear_f_prime = @tanhPrime;
 
  % rand seed
  if params.isGradCheck || params.isProfile
    s = RandStream('mt19937ar','Seed',1);
    RandStream.setGlobalStream(s);
  end
  
  % check GPUs
  params.isGPU = 0;
  if ismac==0
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf('# %d GPUs exist. So, we will use GPUs.\n', n);
      params.isGPU = 1;
      for ii=1:n
        gpuDevice(ii)
      end
    end
  end
  
  % grad check
  if params.isGradCheck
    params.initRange = 0.1;
    params.lstmSize = 2;
    params.batchSize = 10;
  end
  
  % load vocab
  srcVocab = {};
  if params.isGradCheck
    tgtVocab = {'a', 'b'};
    if params.isBi
      srcVocab = {'x', 'y'};
    end
  else
    [tgtVocab] = loadVocab(params.tgtVocabFile);    
    if params.isBi
      [srcVocab] = loadVocab(params.srcVocabFile);
    end
  end
  
  %% add special symbols to vocabs
  if params.isBi
    fprintf(2, '## Bilingual setting\n');
    srcVocab{end+1} = '<eos>';
    params.srcEos = length(srcVocab);
    srcVocab{end+1} = '<sos>';
    params.srcSos = length(srcVocab);
    params.srcVocabSize = length(srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
    tgtVocab{end+1} = '<sos>';
    params.tgtSos = length(tgtVocab);
  end
  tgtVocab{end+1} = '<eos>';
  params.tgtEos = length(tgtVocab);
  params.tgtVocabSize = length(tgtVocab);
  
  %% INIT/LOAD MODEL PARAMETERS %%
  modelFile = [outDir '/model.mat'];
  if params.isGradCheck==0 && exist(modelFile, 'file') % model exists
    fprintf(2, '# Model file %s exists. Try loading ...\n', modelFile);
    savedData = load(modelFile);
    
    % params
    oldParams = savedData.params;
    params.inVocabSize = oldParams.inVocabSize;
    params.outVocabSize = oldParams.outVocabSize;
    params.lr = oldParams.lr;
    params.epoch = oldParams.epoch;
    params.bestCostValid = oldParams.bestCostValid;
    params.testPerplexity = oldParams.testPerplexity;
    startIter = oldParams.iter;
    if params.epoch > 1
      params.iter = (params.epoch-1)*params.epochBatchCount;
    else
      params.iter = 0;  % number of batches we have processed
    end
    
    % model
    model = savedData.model;
    clear savedData;
    
    fprintf(2, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g\n', params.lr, params.epoch, startIter, params.bestCostValid, params.testPerplexity);
  else % start from scratch
    [model, params] = initLSTM(params);
    params.lr = params.learningRate;
    params.epoch = 1;
    params.bestCostValid = 1e5;
    startIter = 0;
    params.iter = 0;  % number of batches we have processed
  end
  
  
  printParams(params);

  %% CHECK GRAD %%
  if params.isGradCheck
    gradCheck(model, params);
    return;
  end
  
  %% load valid & test data
  [inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, numValidWords] = loadPrepareData(params, params.validPrefix, srcVocab, tgtVocab);
  [inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, numTestWords] = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
  %% load train data
  tgtTrainFile = sprintf('%s.%s', params.trainPrefix, params.tgtLang);
  if params.isBi
    srcTrainFile = sprintf('%s.%s', params.trainPrefix, params.srcLang);
    fprintf(2, '# Load train data srcFile "%s" and tgtFile "%s"\n', srcTrainFile, tgtTrainFile);
    srcID = fopen(srcTrainFile, 'r');
    [srcTrainSents] = loadBatchData(srcID, params.baseIndex, params.chunkSize, params.srcEos);
    printSent(srcTrainSents{1}, srcVocab, '  src:');
  else
    fprintf(2, '# Load train data tgtFile "%s"\n', tgtTrainFile);
  end
  tgtID = fopen(tgtTrainFile, 'r');
  [tgtTrainSents, numTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
  printSent(tgtTrainSents{1}, tgtVocab, '  tgt:');

  %%% TRAINING %%%
  totalCost = 0; totalWords = 0;
  evalFreq = params.logFreq*10;
  
  % profile
  if params.isProfile
    profile on
  end
  
  startTime = clock;
  fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  params.epochBatchCount = 0;
  while(1)
    assert(numTrainSents>0);
    numBatches = floor((numTrainSents-1)/params.batchSize) + 1;
    for batchId = 1 : numBatches
      params.iter = params.iter + 1;
      if params.iter <= startIter
        continue;
      end
      startId = (batchId-1)*params.batchSize+1;
      endId = batchId*params.batchSize;
      if endId > numTrainSents
        endId = numTrainSents;
      end
      
      % prepare data
      if params.isBi
        srcBatchSents = srcTrainSents(startId:endId);
      else
        srcBatchSents = {};
      end
      tgtBatchSents = tgtTrainSents(startId:endId);
      [inputTrain, inputTrainMask, tgtTrainOutput, tgtTrainMask, srcTrainMaxLen, tgtTrainMaxLen] = prepareData(srcBatchSents, tgtBatchSents, params);
      
      % core part
      [cost, grad] = lstmCostGrad(model, inputTrain, inputTrainMask, tgtTrainOutput, tgtTrainMask, srcTrainMaxLen, tgtTrainMaxLen, params, 0);
      if isnan(cost) || isinf(cost)
        fprintf(2, 'epoch=%d, iter=%d, nan/inf cost=%g\n', params.epoch, params.iter, cost);
        continue;
      end
      
      % check grad norm
      gradNorm = sqrt(sum(sum(grad.W_emb.^2)) + double(sum(sum(grad.W_soft.^2))) + double(sum(sum(grad.W_src.^2))) + double(sum(sum(grad.W_tgt.^2))));
      scale = 1.0;
      if gradNorm>params.maxGradNorm
        scale = params.maxGradNorm/gradNorm;
      end
      
      % update parameters
      model.W_soft = model.W_soft - params.lr*scale*grad.W_soft;
      if params.isBi
        model.W_src = model.W_src - params.lr*scale*grad.W_src;
      end
      model.W_tgt = model.W_tgt - params.lr*scale*grad.W_tgt;
      indices = find(any(grad.W_emb)); % find out non empty columns
      model.W_emb(:, indices) = model.W_emb(:, indices) - params.lr*scale*grad.W_emb(:, indices);

      % log info
      totalWords = totalWords + sum(sum(tgtTrainMask));
      totalCost = totalCost + cost;
      if mod(params.iter, params.logFreq) == 0
        endTime = clock;
        timeElapsed = etime(endTime, startTime);
        costTrain = totalCost/totalWords;
        fprintf(2, 'epoch=%d, iter=%d, wps=%.2fK, cost=%g, gradNorm=%.2f, srcMaxLen=%d, tgtMaxLen=%d, %.2fs, %s\n', params.epoch, params.iter, totalWords*0.001/timeElapsed, costTrain, gradNorm, srcTrainMaxLen, tgtTrainMaxLen, timeElapsed, datestr(now));
    
        % eval
        if mod(params.iter, evalFreq) == 0    
          if params.isProfile
            if ismac
              profile viewer    
            else
              profsave(profile('info'), 'profile_results');
            end
            return;
          end
          
          costValid = lstmCostGrad(model, inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, params, 1);
          costTest = lstmCostGrad(model, inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, params, 1);
          costValid = costValid/numValidWords;
          costTest = costTest/numTestWords;
          fprintf(2, '# eval %.2f, %d, %d, %.2fK, costTrain=%g, costValid=%g, costTest=%g, %.2fs, %s\n', exp(costTest), params.epoch, params.iter, totalWords*0.001/timeElapsed, costTrain, costValid, costTest, datestr(now));
          
          if costValid < params.bestCostValid
            params.bestCostValid = costValid;
            params.testPerplexity = exp(costTest);
            fprintf(2, '  save model test perplexity %.2f to %s\n', params.testPerplexity, modelFile);
            save(modelFile, 'model', 'params');
          end
        end
        
        % reset
        totalWords = 0;
        totalCost = 0;
        startTime = clock;
      end
    end % end for batchId

    if params.epoch==1
      params.epochBatchCount = params.epochBatchCount + numBatches;
    end
    
    % read more data
    [srcTrainSents, numTrainSents] = loadBatchData(srcID, params.baseIndex, params.chunkSize, params.srcEos);
    [tgtTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
    if numTrainSents == 0 % eof, end of an epoch
      fclose(tgtID);
      fclose(srcID);
      if params.epoch==1
        params.finetuneCount = floor(params.epochFraction*params.epochBatchCount);
        fprintf(2, '# Num batches per epoch = %d, finetune count=%d\n', params.epochBatchCount, params.finetuneCount);
      end
      
      % new epoch
      params.epoch = params.epoch + 1;
      if params.epoch <= params.numEpoches % continue training
        % finetuning
        if params.epoch > params.finetuneEpoch && mod(params.iter, params.finetuneCount)==0
          fprintf(2, '# Finetuning %f -> %f\n', params.lr, params.lr/2);
          params.lr = params.lr/2;
        end
        fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
        
        
        % reopen file
        srcID = fopen(srcTrainFile, 'r');
        tgtID = fopen(tgtTrainFile, 'r');
        [srcTrainSents, numTrainSents] = loadBatchData(srcID, params.baseIndex, params.chunkSize, params.srcEos);
        [tgtTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
      else % done training
        fprintf(2, '# Done training, %s\n', datestr(now));
        break; 
      end
    end
  end % end for while(1)
end


%% Init model parameters %%
function [model, params] = initLSTM(params)
  % special zero words at the end of each vocab
  % in which the emb is all zero and we will never back prob
  % stack vocab:  tgt-vocab + src-vocab
  if params.isBi
    params.inVocabSize = params.tgtVocabSize + params.srcVocabSize;
    model.W_src = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU);
  else
    params.inVocabSize = params.tgtVocabSize;
  end
  params.outVocabSize = params.tgtVocabSize;
  
  model.W_tgt = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU);
  model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize]); % no GPU support for embedding matrix
  model.W_soft = randomMatrix(params.initRange, [params.outVocabSize, params.lstmSize], params.isGPU); % softmax params
  
  % set parameters correspond to zero words
  if params.isBi
    model.W_emb(:, params.tgtVocabSize + params.srcSos) = zeros(params.lstmSize, 1);
  end
  model.W_emb(:, params.tgtEos) = zeros(params.lstmSize, 1);
end


function [input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, numWords] = loadPrepareData(params, ...
    prefix, srcVocab, tgtVocab)  
  if params.isBi % bi
    % src
    srcFile = sprintf('%s.%s', prefix, params.srcLang);
    fprintf(2, '# Loaded data srcFile %s\n', srcFile);
    [srcSents] = loadMonoData(srcFile, params.srcEos, -1, params.baseIndex);
    printSent(srcSents{1}, srcVocab, '  src:');
  else
    srcSents = {};
  end
  
  % tgt
  tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
  fprintf(2, '# Loaded data tgtFile %s\n', tgtFile);
  [tgtSents] = loadMonoData(tgtFile, params.tgtEos, -1, params.baseIndex);
  printSent(tgtSents{1}, tgtVocab, '  tgt:');

  % prepare
  [input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen] = prepareData(srcSents, tgtSents, params);
  
  numWords = sum(sum(tgtMask));
  fprintf(2, '  numWords=%d\n', numWords);
end

function [sents, numSents] = loadMonoData(file, eos, numSents, baseIndex)
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents, eos);
  fclose(fid);
end


function printSent(sent, vocab, prefix)
  fprintf(2, '%s', prefix);
  for ii=1:length(sent)
    fprintf(2, ' %s', vocab{sent(ii)}); 
  end
  fprintf(2, '\n');
end


%% Prepare data %%
%  organize data into matrix format and produce masks, add tgtVocabSize to srcSents:
%   input:      numSents * (srcMaxLen+tgtMaxLen-1)
%   tgtOutput: numSents * tgtMaxLen
%   tgtMask  : numSents * tgtMaxLen, indicate where to ignore in the tgtOutput
%  For the monolingual case, each src sent contains a single simple tgtSos,
%   hence srcMaxLen = 1
function [input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen] = prepareData(srcSents, tgtSents, params)
  if params.isBi
    srcZeroId = params.tgtVocabSize + params.srcSos;
    srcMaxLen = max(cellfun(@(x) length(x), srcSents));
  else
    srcZeroId = params.tgtSos;
    srcMaxLen = 1;
  end
  numSents = length(tgtSents);
  tgtMaxLen = max(cellfun(@(x) length(x), tgtSents));
  input = [srcZeroId*ones(numSents, srcMaxLen) params.tgtEos*ones(numSents, tgtMaxLen-1)];
  tgtOutput = params.tgtEos*ones(numSents, tgtMaxLen);
  for ii=1:numSents
    if params.isBi
      srcLen = length(srcSents{ii});
      input(ii, srcMaxLen-srcLen+1:srcMaxLen) = srcSents{ii} + params.tgtVocabSize; % src part
    end
    
    tgtLen = length(tgtSents{ii});
    input(ii, srcMaxLen+1:srcMaxLen+tgtLen-1) = tgtSents{ii}(1:end-1); % tgt part
    tgtOutput(ii, 1:tgtLen) = tgtSents{ii};
  end
  tgtMask = (tgtOutput~=params.tgtEos);
  if params.isBi
    inputMask = (input~=srcZeroId & input~=params.tgtEos);
  else % for mono case, we still learn parameters for the srcZeroId which is tgtSos.
    inputMask = (input~=params.tgtEos);
  end
  
  % the last src symbol needs to be eos for all sentences
  if params.isBi
    assert(length(unique(input(:, srcMaxLen)))==1); 
    srcEos = srcSents{1}(end) + params.tgtVocabSize;
    assert(input(1, srcMaxLen)==srcEos);
  end
end

%% Check gradients %%

%% Load parallel sentences %%
% function [srcSents, tgtSents, srcNumSents] = loadParallelData(srcFile, tgtFile, srcEos, tgtEos, numSents, baseIndex)
%   [srcSents, srcNumSents] = loadMonoData(srcFile, srcEos, numSents, baseIndex);
%   [tgtSents, tgtNumSents] = loadMonoData(tgtFile, tgtEos, numSents, baseIndex);
%   assert(srcNumSents==tgtNumSents);
% end
