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
  addOptional(p,'finetuneRate', 0.5, @isnumeric); % multiply learning rate by this factor each time we finetune
  addOptional(p,'logFreq', 10, @isnumeric); % how frequent (number of batches) we want to log stuffs
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need to specify other input arguments as toy data is automatically generated.
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'isBi', 1, @isnumeric); % isBi=0: mono model, isBi=1: bi (encoder-decoder) model.
  addOptional(p,'isClip', 0, @isnumeric); % isClip=1: clip forward 50, clip backward 1000.
  addOptional(p,'isResume', 0, @isnumeric); % isResume=1: check if a model file exists, continue training from there.
  addOptional(p,'globalOpt', 0, @isnumeric); % globalOpt=0: no global model, 1: avg global model, 2: feedforward global model.
  addOptional(p,'dataType', 'single', @ischar); % Note: use double precision for grad check
  addOptional(p,'lstmOpt', 0, @isnumeric); % lstmOpt=0: basic model, 1: no tanh for c_t.
  addOptional(p,'seed', 0, @isnumeric); % 0: seed based on current clock time, else use the specified seed
  addOptional(p,'gpuDevice', 1, @isnumeric); % choose the gpuDevice to use. 
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
  if params.isGradCheck || params.isProfile || params.seed
    s = RandStream('mt19937ar','Seed',params.seed);
  else
    s = RandStream('mt19937ar','Seed','shuffle');
  end
  RandStream.setGlobalStream(s);
  
  % check GPUs
  params.isGPU = 0;
  if ismac==0
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf('# %d GPUs exist. So, we will use GPUs. Data type = single.\n', n);
      params.isGPU = 1;
      gpuDevice(gpuDevice)
    end
  end
  
  % grad check
  if params.isGradCheck
    params.dataType = 'double';
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
  params.modelFile = [outDir '/model.mat'];
  if params.isGradCheck==0 && params.isResume && exist(params.modelFile, 'file') % a model exists, resume training
    fprintf(2, '# Model file %s exists. Try loading ...\n', params.modelFile);
    savedData = load(params.modelFile);
    
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
    tic
    gradCheck(model, params);
    toc
    return;
  end
  
  %% Load data
  
  % valid & test
  [validData] = loadPrepareData(params, params.validPrefix, srcVocab, tgtVocab);
  [testData] = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
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
  params.evalFreq = params.logFreq*10;
  
  % profile
  if params.isProfile
    profile on
  end
  
  startTime = clock;
  fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  params.epochBatchCount = 0;
  params.logId = fopen([outDir '/log'], 'w');
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
      [trainData.input, trainData.inputMask, trainData.tgtOutput, trainData.srcMaxLen, trainData.tgtMaxLen, trainData.numWords, trainData.srcLens] = prepareData(srcBatchSents, tgtBatchSents, params);
      % core part
      [cost, grad] = lstmCostGrad(model, trainData, params, 0);
      if isnan(cost) || isinf(cost)
        fprintf(2, 'epoch=%d, iter=%d, nan/inf cost=%g\n', params.epoch, params.iter, cost);
        continue;
      end
      
      %% grad clipping
      gradNorm = double(sum(sum(grad.W_soft.^2))); % sum(sum(grad.W_emb.^2)) + 
      if params.isBi
        for l=1:params.numLayers
          gradNorm = gradNorm + double(sum(sum(grad.W_src{l}.^2)));
        end
      end
      for l=1:params.numLayers
        gradNorm = gradNorm + double(sum(sum(grad.W_tgt{l}.^2)));
      end
      gradNorm = sqrt(gradNorm) / params.batchSize;
      scale = 1.0/params.batchSize; % grad is divided by batchSize
      if gradNorm > params.maxGradNorm
        scale = scale*params.maxGradNorm/gradNorm;
      end
      
      %% update parameters
      model.W_soft = model.W_soft - params.lr*scale*grad.W_soft;
      if params.isBi
        for l=1:params.numLayers
          model.W_src{l} = model.W_src{l} - params.lr*scale*grad.W_src{l};
        end
      end
      for l=1:params.numLayers
        model.W_tgt{l} = model.W_tgt{l} - params.lr*scale*grad.W_tgt{l};
      end
      indices = find(any(grad.W_emb)); % find out non empty columns
      model.W_emb(:, indices) = model.W_emb(:, indices) - params.lr*scale*grad.W_emb(:, indices);

      %% log info
      totalWords = totalWords + trainData.numWords; %sum(sum(trainData.tgtMask));
      totalCost = totalCost + cost;
      if mod(params.iter, params.logFreq) == 0
        endTime = clock;
        timeElapsed = etime(endTime, startTime);
        params.costTrain = totalCost/totalWords;
        params.speed = totalWords*0.001/timeElapsed;
        fprintf(2, 'epoch=%d, iter=%d, wps=%.2fK, lr=%g, cost=%g, gradNorm=%.2f, srcMaxLen=%d, tgtMaxLen=%d, %.2fs, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, trainData.srcMaxLen, trainData.tgtMaxLen, timeElapsed, datestr(now));
        fprintf(params.logId, 'epoch=%d, iter=%d, wps=%.2fK, lr=%g, cost=%g, gradNorm=%.2f, srcMaxLen=%d, tgtMaxLen=%d, %.2fs, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, trainData.srcMaxLen, trainData.tgtMaxLen, timeElapsed, datestr(now));
       
        % reset
        totalWords = 0;
        totalCost = 0;
        startTime = clock;
      end

      %% eval
      if mod(params.iter, params.evalFreq) == 0    
        if params.isProfile
          if ismac
            profile viewer    
          else
            profsave(profile('info'), 'profile_results');
          end
          return;
        end
       
        [params] = evalValidTest(model, validData, testData, params);
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
        if params.evalFreq > params.epochBatchCount
          fprintf(2, '! change evalFreq from %d -> %d\n', params.evalFreq, params.epochBatchCount);
          params.evalFreq = params.epochBatchCount;
          [params] = evalValidTest(model, validData, testData, params);
        end
      end
      
      % new epoch
      params.epoch = params.epoch + 1;
      if params.epoch <= params.numEpoches % continue training
        % finetuning
        if params.epoch > params.finetuneEpoch && mod(params.iter, params.finetuneCount)==0
          fprintf(2, '# Finetuning %f -> %f\n', params.lr, params.lr*params.finetuneRate);
          params.lr = params.lr*params.finetuneRate;
        end
        fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
        
        % reopen file
        tgtID = fopen(tgtTrainFile, 'r');
        [tgtTrainSents, numTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
        if params.isBi
          srcID = fopen(srcTrainFile, 'r');
          [srcTrainSents] = loadBatchData(srcID, params.baseIndex, params.chunkSize, params.srcEos);
        end
      else % done training
        fprintf(2, '# Done training, %s\n', datestr(now));
        break; 
      end
    end
  end % end for while(1)
end

function [params] = evalValidTest(model, validData, testData, params)
  [costValid] = evalCost(model, validData, params); % inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, params);
  [costTest] = evalCost(model, testData, params); %inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, params);
  
  costValid = costValid/validData.numWords;
  costTest = costTest/testData.numWords;
  fprintf(2, '# eval %.2f, %d, %d, %.2fK, %g, costTrain=%g, costValid=%g, costTest=%g, %.2fs, %s\n', exp(costTest), params.epoch, params.iter, params.speed, params.lr, params.costTrain, costValid, costTest, datestr(now));
  fprintf(params.logId, '# eval %.2f, %d, %d, %.2fK, %g, costTrain=%g, costValid=%g, costTest=%g, %.2fs, %s\n', exp(costTest), params.epoch, params.iter, params.speed, params.lr, params.costTrain, costValid, costTest, datestr(now));
  
  if costValid < params.bestCostValid
    params.bestCostValid = costValid;
    params.costTest = costTest;
    params.testPerplexity = exp(costTest);
    fprintf(2, '  save model test perplexity %.2f to %s\n', params.testPerplexity, params.modelFile);
    fprintf(params.logId, '  save model test perplexity %.2f to %s\n', params.testPerplexity, params.modelFile);
    save(params.modelFile, 'model', 'params');
  end
end

%% Eval
function [cost] = evalCost(model, data, params) %input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params)
  numSents = size(data.input, 1);
  numBatches = floor((numSents-1)/params.batchSize) + 1;

  cost = 0;
  for batchId = 1 : numBatches
    startId = (batchId-1)*params.batchSize+1;
    endId = batchId*params.batchSize;
    if endId > numSents
      endId = numSents;
    end
    
    trainData.input = data.input(startId:endId, :);
    trainData.inputMask = data.inputMask(startId:endId, :);
    trainData.tgtOutput = data.tgtOutput(startId:endId, :);
    %trainData.tgtMask = data.tgtMask(startId:endId, :);
    trainData.srcMaxLen = data.srcMaxLen;
    trainData.tgtMaxLen = data.tgtMaxLen;
    cost = cost + lstmCostGrad(model, trainData, params, 1);
  end
end

%% Init model parameters
function [model, params] = initLSTM(params)
  % stack vocab:  tgt-vocab + src-vocab
  modelSize = 0;
  if params.isBi
    params.inVocabSize = params.tgtVocabSize + params.srcVocabSize;
    model.W_src = cell(params.numLayers, 1);
    for l=1:params.numLayers
      model.W_src{l} = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
      modelSize = modelSize + numel(model.W_src{l});
    end
  else
    params.inVocabSize = params.tgtVocabSize;
  end
  params.outVocabSize = params.tgtVocabSize;
  
  model.W_tgt = cell(params.numLayers, 1);
  for l=1:params.numLayers
    model.W_tgt{l} = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
    modelSize = modelSize + numel(model.W_tgt{l});
  end
  model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], 0, 'double'); % no GPU support for embedding matrix. use double since later we will subtract sparse grad matrix and Matlab only supports sparse matrix.
  model.W_soft = randomMatrix(params.initRange, [params.outVocabSize, params.lstmSize], params.isGPU, params.dataType); % softmax params
  modelSize = modelSize + numel(model.W_emb);
  modelSize = modelSize + numel(model.W_soft);
  
  % set parameters correspond to zero words
  if params.isBi
    model.W_emb(:, params.tgtVocabSize + params.srcSos) = zeros(params.lstmSize, 1);
  end
  model.W_emb(:, params.tgtEos) = zeros(params.lstmSize, 1);
  
  fprintf(2, '# Model size = %d\n', modelSize);
  params.modelSize = modelSize;
end


function [data] = loadPrepareData(params, prefix, srcVocab, tgtVocab) % [input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, numWords ] 
  if params.isBi % bi
    % src
    srcFile = sprintf('%s.%s', prefix, params.srcLang);
    [srcSents] = loadMonoData(srcFile, params.srcEos, -1, params.baseIndex, srcVocab, 'src');
  else
    srcSents = {};
  end
  
  % tgt
  tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
  [tgtSents] = loadMonoData(tgtFile, params.tgtEos, -1, params.baseIndex, tgtVocab, 'tgt');

  % prepare
  [data.input, data.inputMask, data.tgtOutput, data.srcMaxLen, data.tgtMaxLen, data.numWords] = prepareData(srcSents, tgtSents, params);
  
  fprintf(2, '  numWords=%d\n', data.numWords);
end

function [sents, numSents] = loadMonoData(file, eos, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents, eos);
  fclose(fid);
  printSent(sents{1}, vocab, ['  ', label, ':']);
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
