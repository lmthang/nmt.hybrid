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
  addOptional(p,'isReverse', 0, @isnumeric); % isReverse=1: src data = $prefix.reversed.$srcLang (instead of $prefix.$srcLang)
  addOptional(p,'isResume', 1, @isnumeric); % isResume=1: check if a model file exists, continue training from there.
  addOptional(p,'softmaxDim', 0, @isnumeric); % softmaxDim>0 convert hidden state into an intermediate representation of size softmaxDim before going through the softmax
  addOptional(p,'dataType', 'single', @ischar); % Note: use double precision for grad check
  addOptional(p,'maxSentLen', 51, @isnumeric); % mostly apply to src, used in attention-based models. Usual length is 50 + 1 (for eos)
  addOptional(p,'sortBatch', 0, @isnumeric); % 1: each time we read in 100 batches, we sort sentences by length.
  addOptional(p,'shuffle', 0, @isnumeric); % 1: shuffle training batches
  
  
  %% debugging options
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need input arguments as toy data is automatically generated.
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'debug', 0, @isnumeric); % 0: no debug, 1: debug
  addOptional(p,'assert', 0, @isnumeric); % 0: no sanity check, 1: yes
  addOptional(p,'seed', 0, @isnumeric); % 0: seed based on current clock time, else use the specified seed

  %% research options
  addOptional(p,'lstmOpt', 0, @isnumeric); % lstmOpt=0: basic model, 1: no tanh for c_t.
  addOptional(p,'gradNormOpt', 0, @isnumeric); % gradOpt=0: basic, 1: add W_emb
  % attnFunc=0: no attention.
  %          1: a_t = softmax(W_a * [tgt_h_t; srcLens]) 
  %          2: a_t = softmax(tanh(W_a * [tgt_h_t; srcLens])) 
  addOptional(p,'attnFunc', 0, @isnumeric);
  addOptional(p,'attnSize', 0, @isnumeric); % dim of the vector used to input to the final softmax, if 0, use lstmSize
  
  addOptional(p,'dropout', 1, @isnumeric); % dropout prob: 1 no dropout, <1: dropout
  
  % positional models: predict pos, then word, use a separate softmax for pos
  % 1: separately print out pos/word perplexities
  % 2: like 1 + feed in src hidden states, 3: like 1 + feed in src embeddings 
  addOptional(p,'posModel', 0, @isnumeric); 
  addOptional(p,'posWin', 7, @isnumeric);
  addOptional(p,'posSoftmax', 0, @isnumeric); % use with posModel. 0: same softmax for word/pos, 1: separate softmax for positions

  addOptional(p,'globalOpt', 0, @isnumeric); % globalOpt=0: no global model, 1: avg global model, 2: feedforward global model.
  
  %% system options
  addOptional(p,'embCPU', 0, @isnumeric); % 1: put W_emb on CPU even if GPUs exist
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
 
  % decode params
  params.beamSize = 12;
  params.stackSize = 100;
  params.unkPenalty = 0;
  params.lengthReward = 0;
  
  % params assertions
  if params.posModel>0
    assert(params.isBi==1);
    assert(params.dropout==1);
  end
  if params.softmaxDim>0
    assert(params.attnFunc==0 & params.posModel==0, '! Assert failed: softmaxDim %d > 0, so attnFunc %d and posModel %d have to be 0.\n', params.softmaxDim, params.attnFunc, params.posModel);
  end

  % rand seed
  if params.isGradCheck || params.isProfile || params.seed
    s = RandStream('mt19937ar','Seed',params.seed);
  else
    s = RandStream('mt19937ar','Seed','shuffle');
  end
  s
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
    params.maxSentLen = 5;
    %params.initRange = 10;
  end
  
  % attention
  if params.attnFunc>0 && params.attnSize==0
    params.attnSize = params.lstmSize;
  end
  assert(strcmp(outDir, '')==0);
  params.logId = fopen([outDir '/log'], 'a');
  
  %% Load vocabs
  [srcVocab, tgtVocab, params] = loadBiVocabs(params);
  
  %% Init / Load Model Parameters
  params.modelFile = [outDir '/model.mat']; % store those with the best valid perplexity
  params.modelRecentFile = [outDir '/modelRecent.mat'];
  [model, params] = initLoadModel(params);
  printParams(1, params);
  printParams(params.logId, params);

  %% Check Grad
  if params.isGradCheck
    tic
    gradCheck(model, params);
    toc
    return;
  end
  
  %%%%%%%%%%%%%%%
  %% Load data %%
  %%%%%%%%%%%%%%%
  % valid & test
  [validData] = loadPrepareData(params, params.validPrefix, srcVocab, tgtVocab);
  [testData] = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
  % train
  params.tgtTrainFile = sprintf('%s.%s', params.trainPrefix, params.tgtLang);
  if params.isBi
    if params.isReverse
      params.srcTrainFile = sprintf('%s.reversed.%s', params.trainPrefix, params.srcLang);
    else
      params.srcTrainFile = sprintf('%s.%s', params.trainPrefix, params.srcLang);
    end
    fprintf(2, '# Load train data srcFile "%s" and tgtFile "%s"\n', params.srcTrainFile, params.tgtTrainFile);
    params.srcTrainId = fopen(params.srcTrainFile, 'r');
  else
    fprintf(2, '# Load train data tgtFile "%s"\n', params.tgtTrainFile);
  end
  params.tgtTrainId = fopen(params.tgtTrainFile, 'r');
  
  
  [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params);
  
  % print
  if params.isBi
    printSent(2, srcTrainSents{1}, srcVocab, '  src 1:');
    printSent(2, srcTrainSents{end}, srcVocab, '  src end:');
  end
  printSent(2, tgtTrainSents{1}, tgtVocab, '  tgt:');
  printSent(2, tgtTrainSents{end}, tgtVocab, '  tgt end:');

  printSent(2, trainBatches{1}.input(1, :), params.vocab, '   input 1:');
  printSent(2, trainBatches{1}.tgtOutput(1, :), params.vocab, '  output 1:');
  % positional models
  if params.posModel>0
    printSent(2, trainBatches{1}.srcPos(1, :), params.vocab, '  srcPos 1:');
  end
  
  %%%%%%%%%%%%%%
  %% Training %%
  %%%%%%%%%%%%%%
  trainCost.total = 0; totalWords = 0;
  if params.posModel>0 % positional model
    trainCost.pos = 0;
    trainCost.word = 0;
  end
  params.evalFreq = params.logFreq*10;
  %params.saveFreq = params.evalFreq;

  % profile
  if params.isProfile
    profile on
  end
  
  startTime = clock;
  fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  fprintf(params.logId, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  isRun = 1;
  lastNanIter = -1;
  nanCount = 0;
  while(isRun)
    assert(numTrainSents>0 && numBatches>0);
    for batchId = 1 : numBatches
      params.iter = params.iter + 1;
      params.batchId = batchId;
      if params.iter <= params.startIter % skip until we readh the point to resume
        continue;
      end
      
      %%%%%%%%%%%%%%%
      %% core part %%
      %%%%%%%%%%%%%%%
      trainData = trainBatches{batchId};
      [costs, grad] = lstmCostGrad(model, trainData, params, 0);

      %% handle nan/inf
      if isnan(costs.total) || isinf(costs.total)
        modelStr = wInfo(model);
        gradStr = wInfo(grad);
        fprintf(2, 'epoch=%d, iter=%d, nan/inf cost=%g, gradStr=%s, modelStr=%s\n', params.epoch, params.iter, costs.total, gradStr, modelStr);
        fprintf(params.logId, 'epoch=%d, iter=%d, nan/inf cost=%g, gradStr=%s, modelStr=%s\n', params.epoch, params.iter, costs.total, gradStr, modelStr);
        if lastNanIter == (params.iter-1) % consecutive nan
          nanCount = nanCount + 1;
        else
          nanCount = 1;
        end
        lastNanIter = params.iter;

        if nanCount==10 % enough patience, stop!
          isRun = 0;
          break;
        else
          continue;
        end
      end
      
      %% grad clipping      
      [gradNorm, ~] = computeGradNorm(grad, params.batchSize, params.varsNoEmb); % historical reason: we exclude W_emb % indNorms
      scale = 1.0/params.batchSize; % grad is divided by batchSize
      if gradNorm > params.maxGradNorm
        scale = scale*params.maxGradNorm/gradNorm;
      end
      scaleLr = params.lr*scale;
      
      %% update parameters
      for ii=1:length(params.varsNoEmb)
        field = params.varsNoEmb{ii};
        if iscell(model.(field))
          for jj=1:length(model.(field)) % cell, like W_src, W_tgt
            model.(field){jj} = model.(field){jj} - scaleLr*grad.(field){jj};
          end
        else
          model.(field) = model.(field) - scaleLr*grad.(field);
        end
      end
      % update W_emb separately
      if params.embCPU && params.isGPU
        model.W_emb(:, grad.indices) = model.W_emb(:, grad.indices) - gather(scaleLr)*grad.W_emb;
      else
        model.W_emb(:, grad.indices) = model.W_emb(:, grad.indices) - scaleLr*grad.W_emb;
      end
      
      %% log info
      totalWords = totalWords + trainData.numWords;
      trainCost.total = trainCost.total + costs.total;
      if params.posModel>0 % positional model
        trainCost.pos = trainCost.pos + costs.pos;
        trainCost.word = trainCost.word + costs.word;
      end
      if mod(params.iter, params.logFreq) == 0
        endTime = clock;
        timeElapsed = etime(endTime, startTime);
        params.costTrain = trainCost.total/totalWords;
        params.speed = totalWords*0.001/timeElapsed;
        if params.posModel>0 % positional model
          params.costTrainPos = trainCost.pos*2/totalWords;
          params.costTrainWord = trainCost.word*2/totalWords;
          fprintf(2, '%d, %d, %.2fK, %g, %.2f (%.2f, %.2f), gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, params.costTrainPos, params.costTrainWord, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
          fprintf(params.logId, '%d, %d, %.2fK, %g, %.2f (%.2f, %.2f), gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, params.costTrainPos, params.costTrainWord, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
        else
          fprintf(2, '%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
          fprintf(params.logId, '%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
        end
        
        % reset
        totalWords = 0;
        trainCost.total = 0;
        if params.posModel>0 % positional model
          trainCost.pos = 0;
          trainCost.word = 0;
        end
        startTime = clock;
      end

      %% eval
      if mod(params.iter, params.evalFreq) == 0    
        % profile
        if params.isProfile
          if ismac
            profile viewer;
          else
            profile off;
            profsave(profile('info'), 'profile_results');
          end
          return;
        end
        
        % eval, save, and decode
        [params] = evalSaveDecode(model, validData, testData, params, srcTrainSents, tgtTrainSents);
        
        startTime = clock;
      end

      % finetuning
      if params.epoch > params.finetuneEpoch && mod(params.iter, params.finetuneCount)==0
        fprintf(2, '# Finetuning %f -> %f\n', params.lr, params.lr*params.finetuneRate);
        fprintf(params.logId, '# Finetuning %f -> %f\n', params.lr, params.lr*params.finetuneRate);
        params.lr = params.lr*params.finetuneRate;
      end
      
      if params.epoch==1
        params.epochBatchCount = params.epochBatchCount + 1;
      end
    end % end for batchId

    % read more data
    [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params);
    
    % eof, end of an epoch
    if numTrainSents == 0 
      % seek to the beginning
      if params.isBi
        fseek(params.srcTrainId, 0, 'bof');
      end
      fseek(params.tgtTrainId, 0, 'bof');

      % read more data
      [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params);
        
      % epoch stats
      if params.epoch==1
        params.finetuneCount = floor(params.epochFraction*params.epochBatchCount);
        fprintf(2, '# Num batches per epoch = %d, finetune count=%d\n', params.epochBatchCount, params.finetuneCount);
        fprintf(params.logId, '# Num batches per epoch = %d, finetune count=%d\n', params.epochBatchCount, params.finetuneCount);
        if params.evalFreq > params.epochBatchCount
          fprintf(2, '! change evalFreq from %d -> %d\n', params.evalFreq, params.epochBatchCount);
          params.evalFreq = params.epochBatchCount;
          
          % eval, save, and decode
          [params] = evalSaveDecode(model, validData, testData, params, srcTrainSents, tgtTrainSents);
        end
      end
      
      % new epoch
      params.epoch = params.epoch + 1;
      if params.epoch <= params.numEpoches % continue training
        fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
        fprintf(params.logId, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
      else % done training
        fprintf(2, '# Done training, %s\n', datestr(now));
        % close files
        if params.isBi
          fclose(params.srcTrainId);
        end
        fclose(params.tgtTrainId);
        break; 
      end
    end
  end % end for while(1)
  
  fclose(params.logId);
end

%% Init model parameters
function [model, params] = initLSTM(params)
  fprintf(2, '# Init LSTM parameters using dataType=%s, initRange=%g\n', params.dataType, params.initRange);
  
  % stack vocab:  tgt-vocab + src-vocab
  if params.isBi
    params.inVocabSize = params.tgtVocabSize + params.srcVocabSize;
    model.W_src = cell(params.numLayers, 1);    
    
    for ll=1:params.numLayers
      model.W_src{ll} = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
  else
    params.inVocabSize = params.tgtVocabSize;
  end
  params.outVocabSize = params.tgtVocabSize;
  
  % W_tgt
  model.W_tgt = cell(params.numLayers, 1);
  for ll=1:params.numLayers
    if ll==1 && params.posModel>0 % W_tgt [x_t; h_t; s_t] where s_t is the source info hinted by the positional model
      model.W_tgt{ll} = randomMatrix(params.initRange, [4*params.lstmSize, 3*params.lstmSize], params.isGPU, params.dataType);
    else
      model.W_tgt{ll} = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
  end
 
  % W_emb
  if params.embCPU == 1
    fprintf(2, '# W_emb is explicitly put on CPU\n');
    model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], 0, 'double');
  else
    model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], params.isGPU, params.dataType);
  end
  % set parameters correspond to zero words
  if params.isBi
    model.W_emb(:, params.tgtVocabSize + params.srcSos) = zeros(params.lstmSize, 1);
  end
  model.W_emb(:, params.tgtEos) = zeros(params.lstmSize, 1);
  
  % attention mechanism
  if params.attnFunc>0 
    model.W_a = randomMatrix(params.initRange, [params.maxSentLen, params.lstmSize+1], params.isGPU, params.dataType);
    % attn_t = H_src * a_t
    % h_attn_t = f(W_ah * [attn_t; h_t])
    model.W_ah = randomMatrix(params.initRange, [params.attnSize, 2*params.lstmSize], params.isGPU, params.dataType);
    
    params.softmaxSize = params.attnSize;
  else
    params.softmaxSize = params.lstmSize;
  end
  
  % compress softmax
  if params.softmaxDim>0 
    model.W_h = randomMatrix(params.initRange, [params.softmaxDim, params.lstmSize], params.isGPU, params.dataType);
    params.softmaxSize = params.softmaxDim;
  end
  
  % W_soft
  model.W_soft = randomMatrix(params.initRange, [params.outVocabSize, params.softmaxSize], params.isGPU, params.dataType);
  
  % positional models
  if params.posModel>0
    model.W_softPos = randomMatrix(params.initRange, [params.posVocabSize, params.softmaxSize], params.isGPU, params.dataType);
  end
  
  % compute model size
  params.modelSize = modelSizes(model);
end

function [params] = evalSaveDecode(model, validData, testData, params, srcTrainSents, tgtTrainSents)
  % eval
  [params] = evalValidTest(model, validData, testData, params);

  % save
  fprintf(2, '  save model cur test perplexity %.2f to %s\n', params.curTestPerplexity, params.modelRecentFile);
  fprintf(params.logId, '  save model cur test perplexity %.2f to %s\n', params.curTestPerplexity, params.modelRecentFile);
  save(params.modelRecentFile, 'model', 'params');

  % decode
  if params.isBi && params.posModel==0
    validId = randi(validData.numSents;
    testId = randi(testData.numSents);
    srcDecodeSents = [srcTrainSents(1); validData.srcSents(validId)); testData.srcSents(testId)];
    tgtDecodeSents = [tgtTrainSents(1); validData.tgtSents(validId); testData.tgtSents(testId)];
    [decodeData] = prepareData(srcDecodeSents, tgtDecodeSents, 1, params);
    decodeData.startId = 1;
    [candidates, candScores] = lstmDecoder(model, decodeData, params);
    printDecodeResults(decodeData, candidates, candScores, params, 0);
  end
end

function [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params)
  if params.isBi
    [srcTrainSents, ~, srcTrainLens] = loadBatchData(params.srcTrainId, params.baseIndex, params.chunkSize, params.srcEos);
  else
    srcTrainSents = {};
  end
  [tgtTrainSents, numTrainSents, tgtTrainLens] = loadBatchData(params.tgtTrainId, params.baseIndex, params.chunkSize, params.tgtEos);
  
  % sorting
  if params.sortBatch
    [tgtTrainLens, sortIndices] = sort(tgtTrainLens);
    tgtTrainSents = tgtTrainSents(sortIndices);
    if params.isBi
      srcTrainSents = srcTrainSents(sortIndices);
      srcTrainLens = srcTrainLens(sortIndices);
    end
  end
  
  % split into batches
  if numTrainSents>0
    numBatches = floor((numTrainSents-1)/params.batchSize) + 1;
    trainBatches = cell(numBatches, 1);
    
    srcBatchSents = {};
    srcBatchLens = [];
    for batchId = 1 : numBatches
      startId = (batchId-1)*params.batchSize+1;
      endId = batchId*params.batchSize;
      if endId > numTrainSents
        endId = numTrainSents;
      end
      
      % prepare data
      if params.isBi
        srcBatchSents = srcTrainSents(startId:endId);
        srcBatchLens = srcTrainLens(startId:endId);
      end
      tgtBatchSents = tgtTrainSents(startId:endId);
      tgtBatchLens = tgtTrainLens(startId:endId);
      trainBatches{batchId} = prepareData(srcBatchSents, tgtBatchSents, 0, params, srcBatchLens, tgtBatchLens);
    end
    
    % shuffle
    if params.shuffle
      trainBatches = trainBatches(randperm(numBatches));
    end
  else
    trainBatches = {};
    numBatches = 0;
  end
end

function [model, params] = initLoadModel(params)
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

    params.startIter = oldParams.iter;
    if params.epoch > 1
      params.iter = (params.epoch-1)*params.epochBatchCount;
    else
      params.iter = 0;  % number of batches we have processed
    end
    
    % model
    model = savedData.model;
    clear savedData;
    
    fprintf(2, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g\n', params.lr, params.epoch, params.startIter, params.bestCostValid, params.testPerplexity);
    fprintf(params.logId, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g\n', params.lr, params.epoch, params.startIter, params.bestCostValid, params.testPerplexity);
  else % start from scratch
    [model, params] = initLSTM(params);
    params.lr = params.learningRate;
    params.epoch = 1;
    params.bestCostValid = 1e5;
    params.testPerplexity = 1e5;
    params.curTestPerplexity = 1e5;
    params.startIter = 0;
    params.iter = 0;  % number of batches we have processed
    params.epochBatchCount = 0;
    params.finetuneCount = 0;
  end

  params = setupVars(model, params);
end


function [params] = setupVars(model, params)
  params.vars = fields(model);
  
  % right now, we have been training models in which the gradNorm
  % computation excludes W_emb, so we'll stick with that for now.
  params.varsNoEmb = params.vars;
  for ii=1:length(params.vars)
    if strcmp(params.vars{ii}, 'W_emb')
      params.varsNoEmb(ii) = [];
      break;
    end
  end
end

function [srcVocab, tgtVocab, params] = loadBiVocabs(params)
  srcVocab = {};
  if params.isGradCheck
    if params.posModel>0
      params.posWin = 2;
      tgtVocab = {'a', 'b', '<p_-2>', '<p_-1>', '<p_0>', '<p_1>', '<p_2>', '<p_n>'};
    else
      tgtVocab = {'a', 'b'};
    end
    
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
    
    %% positional vocab
    if params.posModel>0
      params.posVocabSize = 2*params.posWin + 3; % for W_softPos, -posWin, ..., 0, posWin, p_n, p_eos
      params.startPosId = length(tgtVocab) - 2*params.posWin - 1; % marks -posWin
      params.zeroPosId = params.startPosId + params.posWin;
      
      % assert: -posWin, ..., 0, posWin, p_n are those last words in the tgtVocab
      assert(params.startPosId == (length(srcVocab)+1));
      for ii=1:(2*params.posWin +1)
        relPos = ii-params.posWin-1;
        assert(strcmp(tgtVocab{params.startPosId + ii - 1}, ['<p_', num2str(relPos), '>'])==1);
      end
      % p_n
      params.nullPosId = length(tgtVocab);
      assert(strcmp(tgtVocab{params.nullPosId}, '<p_n>')==1);
      % p_eos
      tgtVocab{end+1} = '<p_eos>';
      params.eosPosId = length(tgtVocab);
      
      fprintf(2, '# Positional model: zeroPosId %s=%d, nullPosId %s=%d, eosPosId %s=%d\n', tgtVocab{params.zeroPosId}, params.zeroPosId, tgtVocab{params.nullPosId}, params.nullPosId, tgtVocab{params.eosPosId}, params.eosPosId);
      fprintf(params.logId, '# Positional model: zeroPosId %s=%d, nullPosId %s=%d, eosPosId %s=%d\n', tgtVocab{params.zeroPosId}, params.zeroPosId, tgtVocab{params.nullPosId}, params.nullPosId, tgtVocab{params.eosPosId}, params.eosPosId);
    end
    
    %% append src vocab
    srcVocab{end+1} = '<s_eos>';
    params.srcEos = length(srcVocab);
    srcVocab{end+1} = '<s_sos>';
    params.srcSos = length(srcVocab);
    % here we have src eos, so we don't need tgt sos.
    params.srcVocabSize = length(srcVocab);
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
end

function [data] = loadPrepareData(params, prefix, srcVocab, tgtVocab)
  [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab);
  [data] = prepareData(srcSents, tgtSents, 1, params);
  fprintf(2, '  numSents=%d, numWords=%d\n', numSents, data.numWords);
  data.numSents = numSents;
  data.srcSents = srcSents;
  data.tgtSents = tgtSents;
end

%% Unused code %%
  % attnOpt=0: no attention.
  %         1: bilingual embedding attention.
  %         2: same as Bengio. NOT IMPLEMENTED.
  %addOptional(p,'attnOpt', 0, @isnumeric); 
  % assert(params.attnOpt==0 || params.attnFunc>0);
  % assert(params.attnFunc==0 || params.attnOpt>0);
  %         1: a_t,i = f(tgt_h_t, src_h_i) = tanh(tgt_h_t' * W_a * src_h_i)
  %         2: simialr to Bengio a_t,i = f(tgt_h_t, src_h_i) = v_a'tanh(W_a_tgt*tgt_h_t + W_a_src*src_h_i). NOT IMPLEMENTED.if params.attnFunc==1 % tanh(tgt_h_t' * W_a * src_h_i)
%     model.W_a = randomMatrix(params.initRange, [params.lstmSize, params.lstmSize], params.isGPU, params.dataType);
%   elseif params.attnFunc==2 % v_a'tanh(W_a_tgt*tgt_h_t + W_a*src_h_i)
%     model.W_a = randomMatrix(params.initRange, [params.lstmSize, params.lstmSize], params.isGPU, params.dataType);
%     model.W_a_tgt = randomMatrix(params.initRange, [params.lstmSize, params.lstmSize], params.isGPU, params.dataType);
%     model.v_a = randomMatrix(params.initRange, [params.lstmSize, 1], params.isGPU, params.dataType);
%   else
%   end


%       if params.softmaxDim>0
%         model.W_h = model.W_h - scaleLr*grad.W_h;
%       end
%       model.W_soft = model.W_soft - scaleLr*grad.W_soft;
%       if params.isBi
%         for l=1:params.numLayers
%           model.W_src{l} = model.W_src{l} - scaleLr*grad.W_src{l};
%         end
%       end
%       for l=1:params.numLayers
%         model.W_tgt{l} = model.W_tgt{l} - scaleLr*grad.W_tgt{l};
%       end


%       gradNorm = double(sum(sum(grad.W_soft.^2))); % + sum(sum(grad.W_emb.^2));
%       if params.softmaxDim>0
%         gradNorm = gradNorm + double(sum(sum(grad.W_h.^2)));
%       end
%       if params.isBi
%         for l=1:params.numLayers
%           gradNorm = gradNorm + double(sum(sum(grad.W_src{l}.^2)));
%         end
%       end
%       for l=1:params.numLayers
%         gradNorm = gradNorm + double(sum(sum(grad.W_tgt{l}.^2)));
%       end
%       gradNorm = sqrt(gradNorm) / params.batchSize;
      
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
