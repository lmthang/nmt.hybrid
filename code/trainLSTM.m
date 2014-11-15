%% Train Long-Short Term Memory (LSTM).
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
% Options:
%
%   srcLang, srcVocabFile: leave empty to train monolingual models.
%
%   baseIndex: of training data. Required to convert them to 1-indexed.
%
%%%
function trainLSTM(trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,baseIndex,varargin)  
  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));
  
  %% Argument Parser
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
  addOptional(p,'numLayers', 1, @isnumeric); % number of layers
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
  addOptional(p,'isBi', 1, @isnumeric); % isBi=0: mono model, isBi=1: bi (encoder-decoder) model.
  addOptional(p,'isClip', 0, @isnumeric); % isClip=1: clip forward 50, clip backward 1000.
  addOptional(p,'isResume', 0, @isnumeric); % isResume=1: check if a model file exists, continue training from there.
  addOptional(p,'globalOpt', 0, @isnumeric); % globalOpt=0: no global model, 1: avg global model, 2: feedforward global model.
  p.KeepUnmatched = true;
  parse(p,trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,baseIndex,varargin{:})
  params = p.Results;
  
  %% Setup params
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
  
  %% Load vocabs
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
  
  % add special symbols to vocabs
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
  
  %% Init / Load Model Parameters
  modelFile = [outDir '/model.mat'];
  if params.isGradCheck==0 && params.isResume && exist(modelFile, 'file') % a model exists, resume training
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

  %% Check Grad
  if params.isGradCheck
    gradCheck(model, params);
    return;
  end
  
  %% Load data
  
  % valid & test
  [inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, numValidWords] = loadPrepareData(params, params.validPrefix, srcVocab, tgtVocab);
  [inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, numTestWords] = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
  % train
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

  %% Training
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
      gradNorm = gradNorm / params.batchSize;
      scale = 1.0/params.batchSize; % grad is divided by batchSize
      if gradNorm > params.maxGradNorm
        scale = scale*params.maxGradNorm/gradNorm;
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
        fprintf(2, 'epoch=%d, iter=%d, wps=%.2fK, lr=%g, cost=%g, gradNorm=%.2f, srcMaxLen=%d, tgtMaxLen=%d, %.2fs, %s\n', params.epoch, params.iter, totalWords*0.001/timeElapsed, params.lr, costTrain, gradNorm, srcTrainMaxLen, tgtTrainMaxLen, timeElapsed, datestr(now));
    
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
          
          
          %costValid = lstmCostGrad(model, inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, params, 1);
          %costTest = lstmCostGrad(model, inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, params, 1);
          [costValid] = evalCost(model, inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, params);
          [costTest] = evalCost(model, inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, params);
          
          costValid = costValid/numValidWords;
          costTest = costTest/numTestWords;
          fprintf(2, '# eval %.2f, %d, %d, %.2fK, %g, costTrain=%g, costValid=%g, costTest=%g, %.2fs, %s\n', exp(costTest), params.epoch, params.iter, totalWords*0.001/timeElapsed, params.lr, costTrain, costValid, costTest, datestr(now));
          
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
    [tgtTrainSents, numTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
    if params.isBi
      [srcTrainSents] = loadBatchData(srcID, params.baseIndex, params.chunkSize, params.srcEos);
    end
    if numTrainSents == 0 % eof, end of an epoch
      fclose(tgtID);
      if params.isBi
        fclose(srcID);
      end
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
        tgtID = fopen(tgtTrainFile, 'r');
        [tgtTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
        if params.isBi
          srcID = fopen(srcTrainFile, 'r');
          [srcTrainSents, numTrainSents] = loadBatchData(srcID, params.baseIndex, params.chunkSize, params.srcEos);
        end
      else % done training
        fprintf(2, '# Done training, %s\n', datestr(now));
        break; 
      end
    end
  end % end for while(1)
end

%% Eval
function [cost] = evalCost(model, input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params)
  numSents = size(input, 1);
  numBatches = floor((numSents-1)/params.batchSize) + 1;

  cost = 0;
  for batchId = 1 : numBatches
    startId = (batchId-1)*params.batchSize+1;
    endId = batchId*params.batchSize;
    if endId > numSents
      endId = numSents;
    end
    
    cost = cost + lstmCostGrad(model, input(startId:endId, :), inputMask(startId:endId, :), tgtOutput(startId:endId, :), ...
      tgtMask(startId:endId, :), srcMaxLen, tgtMaxLen, params, 1);
  end
end

%% Init model parameters
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


%% Check gradients %%

%% Load parallel sentences %%
% function [srcSents, tgtSents, srcNumSents] = loadParallelData(srcFile, tgtFile, srcEos, tgtEos, numSents, baseIndex)
%   [srcSents, srcNumSents] = loadMonoData(srcFile, srcEos, numSents, baseIndex);
%   [tgtSents, tgtNumSents] = loadMonoData(tgtFile, tgtEos, numSents, baseIndex);
%   assert(srcNumSents==tgtNumSents);
% end
