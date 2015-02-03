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
  addOptional(p,'isBi', 1, @isnumeric); % isBi=0: mono model, isBi=1: bi (encoder-decoder) model.
  addOptional(p,'isClip', 0, @isnumeric); % isClip=1: clip forward 50, clip backward 1000.
  addOptional(p,'isResume', 1, @isnumeric); % isResume=1: check if a model file exists, continue training from there.
  addOptional(p,'dataType', 'single', @ischar); % Note: use double precision for grad check

  %% debugging options
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need input arguments as toy data is automatically generated.
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'debug', 0, @isnumeric); % 0: no debug, 1: debug
  addOptional(p,'assert', 0, @isnumeric); % 0: no sanity check, 1: yes
  addOptional(p,'seed', 0, @isnumeric); % 0: seed based on current clock time, else use the specified seed

  %% research options
  addOptional(p,'lstmOpt', 0, @isnumeric); % lstmOpt=0: basic model, 1: no tanh for c_t.
  addOptional(p,'attnOpt', 0, @isnumeric); % attnOpt=0: no attention, 1: bilingual embedding attention
  addOptional(p,'globalOpt', 0, @isnumeric); % globalOpt=0: no global model, 1: avg global model, 2: feedforward global model.
  addOptional(p,'f_bias', 0, @isnumeric); % bias added to the forget gate

  %% system options
  addOptional(p,'onlyCPU', 0, @isnumeric); % 1: avoid using GPUs
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
  if ismac==0 && params.onlyCPU==0
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf(2, '# %d GPUs exist. So, we will use GPUs.\n', n);
      params.isGPU = 1;
      %reset(gpuDevice(params.gpuDevice));
      gpuDevice(params.gpuDevice)
    else
      params.dataType = 'double';
    end
  else
    params.dataType = 'double';
  end
  
  % grad check
  if params.isGradCheck
    params.lstmSize = 2;
    params.batchSize = 10;
    params.batchId = 1;
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
    srcVocab{end+1} = '<s_eos>';
    params.srcEos = length(srcVocab);
    srcVocab{end+1} = '<s_sos>';
    params.srcSos = length(srcVocab);
    params.srcVocabSize = length(srcVocab);
    % here we have src eos, so we don't need tgt sos.
  else
    fprintf(2, '## Monolingual setting\n');
    srcVocab = {};
    tgtVocab{end+1} = '<t_sos>';
    params.tgtSos = length(tgtVocab);
  end
  tgtVocab{end+1} = '<t_eos>';
  params.tgtEos = length(tgtVocab);
  params.tgtVocabSize = length(tgtVocab);
  params.vocab = [tgtVocab srcVocab];
  
  assert(strcmp(outDir, '')==0);
    
  params.logId = fopen([outDir '/log'], 'a');
  
  %% Init / Load Model Parameters
  params.modelFile = [outDir '/model.mat']; % store those with the best valid perplexity
  params.modelRecentFile = [outDir '/modelRecent.mat'];
  if params.isGradCheck==0 && params.isResume && exist(params.modelRecentFile, 'file') % a model exists, resume training
    fprintf(2, '# Model file %s exists. Try loading ...\n', params.modelRecentFile);
    fprintf(params.logId, '# Model file %s exists. Try loading ...\n', params.modelRecentFile);
    savedData = load(params.modelRecentFile);
    
    % params
    oldParams = savedData.params;
    params.inVocabSize = oldParams.inVocabSize;
    params.outVocabSize = oldParams.outVocabSize;
    params.lr = oldParams.lr;
    params.epoch = oldParams.epoch;
    params.epochBatchCount = oldParams.epochBatchCount;
    params.bestCostValid = oldParams.bestCostValid;
    params.testPerplexity = oldParams.testPerplexity;
    if isfield(oldParams, 'finetuneCount')
      params.finetuneCount = oldParams.finetuneCount;
    else
      if params.epoch > params.finetuneEpoch && params.epochBatchCount>0 % try to determine finetuneCount, we should rarely need this
        params.finetuneCount = floor(params.epochFraction*params.epochBatchCount);
      else
        params.finetuneCount = 0;
      end
    end

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
    fprintf(params.logId, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g\n', params.lr, params.epoch, startIter, params.bestCostValid, params.testPerplexity);
  else % start from scratch
    [model, params] = initLSTM(params);
    params.lr = params.learningRate;
    params.epoch = 1;
    params.bestCostValid = 1e5;
    startIter = 0;
    params.iter = 0;  % number of batches we have processed
    params.epochBatchCount = 0;
    params.finetuneCount = 0;
  end
  
  printParams(1, params);
  printParams(params.logId, params);

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
    printSent(srcTrainSents{1}, srcVocab, '  src 1:');
    printSent(srcTrainSents{end}, srcVocab, '  src end:');
  else
    fprintf(2, '# Load train data tgtFile "%s"\n', tgtTrainFile);
  end
  tgtID = fopen(tgtTrainFile, 'r');
  [tgtTrainSents, numTrainSents] = loadBatchData(tgtID, params.baseIndex, params.chunkSize, params.tgtEos);
  printSent(tgtTrainSents{1}, tgtVocab, '  tgt:');
  printSent(tgtTrainSents{end}, tgtVocab, '  tgt end:');

  %% Training
  totalCost = 0; totalWords = 0;
  params.evalFreq = params.logFreq*10;
  params.saveFreq = params.evalFreq;

  % profile
  if params.isProfile
    profile on
  end
  
  startTime = clock;
  fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  fprintf(params.logId, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  isRun = 1;
  while(isRun)
    assert(numTrainSents>0);
    numBatches = floor((numTrainSents-1)/params.batchSize) + 1;
    for batchId = 1 : numBatches
      params.iter = params.iter + 1;
      params.batchId = batchId;
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
      
      %vocab = [tgtVocab srcVocab]
      %printSent(trainData.input(1, :), vocab, '  input:');
      if isnan(cost) || isinf(cost)
        fprintf(2, 'epoch=%d, iter=%d, nan/inf cost=%g\n', params.epoch, params.iter, cost);
        fprintf(params.logId, 'epoch=%d, iter=%d, nan/inf cost=%g\n', params.epoch, params.iter, cost);
        isRun = 0;
        break;
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
      
      scaleLr = params.lr*scale;
      %% update parameters
      model.W_soft = model.W_soft - scaleLr*grad.W_soft;
      if params.isBi
        for l=1:params.numLayers
          model.W_src{l} = model.W_src{l} - scaleLr*grad.W_src{l};
        end
      end
      for l=1:params.numLayers
        model.W_tgt{l} = model.W_tgt{l} - scaleLr*grad.W_tgt{l};
      end
      model.W_emb(:, grad.indices) = model.W_emb(:, grad.indices) - scaleLr*grad.W_emb;
      
      %% log info
      totalWords = totalWords + trainData.numWords;
      totalCost = totalCost + cost;
      if mod(params.iter, params.logFreq) == 0
        endTime = clock;
        timeElapsed = etime(endTime, startTime);
        params.costTrain = totalCost/totalWords;
        params.speed = totalWords*0.001/timeElapsed;
        modelStr = wInfo(model);
        fprintf(2, '%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s, s=%d, t=%d, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, modelStr, trainData.srcMaxLen, trainData.tgtMaxLen, datestr(now));
        fprintf(params.logId, '%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s, s=%d, t=%d, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, modelStr, trainData.srcMaxLen, trainData.tgtMaxLen, datestr(now));
        
        % reset
        totalWords = 0;
        totalCost = 0;
        startTime = clock;
      end

      %% eval
      if mod(params.iter, params.evalFreq) == 0    
        if params.isProfile
          if ismac
            profile viewer;
          else
            profile off;
            profsave(profile('info'), 'profile_results');
          end
          return;
        end
       
        [params] = evalValidTest(model, validData, testData, params);
      end

      %% save
      if mod(params.iter, params.saveFreq) == 0    
        fprintf(2, '  save model cur test perplexity %.2f to %s\n', params.curTestPerplexity, params.modelRecentFile);
        fprintf(params.logId, '  save model cur test perplexity %.2f to %s\n', params.curTestPerplexity, params.modelRecentFile);
        save(params.modelRecentFile, 'model', 'params');
      end

      % finetuning
      if params.epoch > params.finetuneEpoch && mod(params.iter, params.finetuneCount)==0
        fprintf(2, '# Finetuning %f -> %f\n', params.lr, params.lr*params.finetuneRate);
        fprintf(params.logId, '# Finetuning %f -> %f\n', params.lr, params.lr*params.finetuneRate);
        params.lr = params.lr*params.finetuneRate;
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
    
    % eof, end of an epoch
    if numTrainSents == 0 
      fclose(tgtID);
      if params.isBi
        fclose(srcID);
      end
      if params.epoch==1
        params.finetuneCount = floor(params.epochFraction*params.epochBatchCount);
        fprintf(2, '# Num batches per epoch = %d, finetune count=%d\n', params.epochBatchCount, params.finetuneCount);
        fprintf(params.logId, '# Num batches per epoch = %d, finetune count=%d\n', params.epochBatchCount, params.finetuneCount);
        if params.evalFreq > params.epochBatchCount
          fprintf(2, '! change evalFreq from %d -> %d\n', params.evalFreq, params.epochBatchCount);
          params.evalFreq = params.epochBatchCount;
          [params] = evalValidTest(model, validData, testData, params);
        end
      end
      
      % new epoch
      params.epoch = params.epoch + 1;
      if params.epoch <= params.numEpoches % continue training
        fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
        fprintf(params.logId, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
        
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
  
  fclose(params.logId);
end

function [params] = evalValidTest(model, validData, testData, params)
  [costValid] = evalCost(model, validData, params); % inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, params);
  [costTest] = evalCost(model, testData, params); %inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, params);
  
  costValid = costValid/validData.numWords;
  costTest = costTest/testData.numWords;
  fprintf(2, '# eval %.2f, %d, %d, %.2fK, %.2f, train=%.4f, valid=%.4f, test=%.4f, %.2fs, %s\n', exp(costTest), params.epoch, params.iter, params.speed, params.lr, params.costTrain, costValid, costTest, datestr(now));
  fprintf(params.logId, '# eval %.2f, %d, %d, %.2fK, %.2f, train=%.4f, valid=%.4f, test=%.4f, %.2fs, %s\n', exp(costTest), params.epoch, params.iter, params.speed, params.lr, params.costTrain, costValid, costTest, datestr(now));
    
  params.curTestPerplexity = exp(costTest);
  
  if costValid < params.bestCostValid
    params.bestCostValid = costValid;
    params.costTest = costTest;
    params.testPerplexity = params.curTestPerplexity;
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
  trainData.srcMaxLen = data.srcMaxLen;
  trainData.tgtMaxLen = data.tgtMaxLen;
  for batchId = 1 : numBatches
    startId = (batchId-1)*params.batchSize+1;
    endId = batchId*params.batchSize;
    if endId > numSents
      endId = numSents;
    end
    
    trainData.input = data.input(startId:endId, :);
    trainData.inputMask = data.inputMask(startId:endId, :);
    trainData.tgtOutput = data.tgtOutput(startId:endId, :);
    cost = cost + lstmCostGrad(model, trainData, params, 1);
  end
end

%% Init model parameters
function [model, params] = initLSTM(params)
  fprintf(2, '# Init LSTM parameters using dataType=%s, initRange=%f\n', params.dataType, params.initRange);
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
  %model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], 0, 'double');
  model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], params.isGPU, params.dataType);
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


function [data] = loadPrepareData(params, prefix, srcVocab, tgtVocab)
  % src
  if params.isBi
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
  printSent(sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(sents{end}, vocab, ['  ', label, ' end:']);
end


%% Check gradients %%

%% Load parallel sentences %%
% function [srcSents, tgtSents, srcNumSents] = loadParallelData(srcFile, tgtFile, srcEos, tgtEos, numSents, baseIndex)
%   [srcSents, srcNumSents] = loadMonoData(srcFile, srcEos, numSents, baseIndex);
%   [tgtSents, tgtNumSents] = loadMonoData(tgtFile, tgtEos, numSents, baseIndex);
%   assert(srcNumSents==tgtNumSents);
% end

      %if params.isGPU
      %  emb_gpu = gpuArray(full(grad.W_emb(:, grad.indices)));
      %  model.W_emb(:, grad.indices) = model.W_emb(:, grad.indices) - scaleLr*emb_gpu;
      %else
      %end

%       for t=1:length(grad.indices)
%         indices = grad.indices{t};
%         emb_grad = grad.emb{t};
%         for jj=1:length(indices)
%           model.W_emb(:, indices(jj)) = model.W_emb(:, indices(jj)) - scaleLr*emb_grad(:, jj);
%         end
%       end
