function trainLSTM(trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,varargin)  
% Train Long-Short Term Memory (LSTM) models.
% Arguments:
%   trainPrefix, validPrefix, testPrefix: expect files trainPrefix.srcLang,
%     trainPrefix.tgtLang. Similarly for validPrefix and testPrefix.
%     These data files contain sequences of integers one per line.
%   srcLang, tgtLang: languages, e.g. en, de.
%   srcVocabFile, tgtVocabFile: one word per line.
%   outDir: output directory.
%   varargin: other optional arguments.
%
% Thang Luong @ 2014, 2015, <lmthang@stanford.edu>
% With contributions from:
%   Hieu Pham: beam-search decoder.

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
  
  % important hyperparameters
  addOptional(p,'numLayers', 1, @isnumeric); % stacking architecture
  addOptional(p,'lstmSize', 100, @isnumeric); % number of cells, also embedding size
  addOptional(p,'initRange', 0.1, @isnumeric);
  addOptional(p,'learningRate', 1.0, @isnumeric);
  addOptional(p,'maxGradNorm', 5.0, @isnumeric); % to scale large grads
  addOptional(p,'batchSize', 128, @isnumeric);
  addOptional(p,'numEpoches', 10, @isnumeric); % num epoches
  addOptional(p,'epochFraction', 1.0, @isnumeric);
  addOptional(p,'finetuneEpoch', 5, @isnumeric); % epoch > finetuneEpoch, start halving learning rate every epochFraction of an epoch, e.g., every 0.5 epoch
  addOptional(p,'finetuneRate', 0.5, @isnumeric); % multiply learning rate by this factor each time we finetune
  
  % hack
  addOptional(p,'epochIter', 0, @isnumeric); % if our train file is too large and we want to know the number of iterations in a epoch beforehand
  
  % advanced features
  addOptional(p,'dropout', 1, @isnumeric); % keep prob for dropout, i.e., 1 no dropout, <1: dropout
  addOptional(p,'isReverse', 0, @isnumeric); % 1: reseverse source sentence. We expect file $prefix.$srcLang.reversed (instead of $prefix.$srcLang)
  addOptional(p,'feedInput', 0, @isnumeric); % 1: feed the softmax vector to the next timestep input
  addOptional(p,'lstmOpt', 0, @isnumeric); % lstmOpt=0: basic model (I have always been using this!), 1: no tanh for c_t.
    
  % training
  addOptional(p,'isBi', 1, @isnumeric); % isBi=0: mono model, isBi=1: bi (encoder-decoder) model.
  addOptional(p,'isClip', 1, @isnumeric); % isClip=1: clip forward 50, clip backward 1000.
  addOptional(p,'maxSentLen', 51, @isnumeric); % limit sentence length on each side during training. Default: 50 + 1 (eos).
  addOptional(p,'logFreq', 10, @isnumeric); % how frequent (number of batches) we want to log stuffs
  addOptional(p,'isResume', 1, @isnumeric); % isResume=1: check if a model file exists, continue training from there.
  addOptional(p,'sortBatch', 1, @isnumeric); % 1: each time we read in 100 batches, we sort sentences by length.
  addOptional(p,'shuffle', 1, @isnumeric); % 1: shuffle training batches
  addOptional(p,'loadModel', '', @ischar); % To start training from
  addOptional(p,'saveHDF', 0, @isnumeric); % 1: to save in HDF5 format
  
  % char-based models
  addOptional(p,'charShortList', 0, @isnumeric); % list of frequent words after which we will learn compositions from characters
  addOptional(p,'charPrefix', '', @ischar); % list of characters
  addOptional(p,'charNumLayers', 1, @isnumeric);
  %addOptional(p,'charMapFile', '', @ischar); % map words into sequences of chars (in integers)
  % trainLSTM('../output/id.shortlist.100/train.10k', '../output/id.shortlist.100/valid.100', '../output/id.shortlist.100/test.100', 'de', 'en', '../output/id.1000/shortlist.100.de.vocab', '../output/id.1000/shortlist.100.en.vocab', '../output/basic', 'isResume', 0, 'charShortList', 100, 'charPrefix', '../output/id.1000/shortlist.100', 'logFreq', 1); 
  %'../output/id.1000/shortlist.100.de.char.vocab', 'tgtCharVocabFile', '../output/id.1000/shortlist.100.en.char.vocab', 'srcCharMapFile', '../output/id.1000/shortlist.100.de.char.map', 'srcCharMapFile', '../output/id.1000/shortlist.100.en.char.map')
  
  % decoding
  addOptional(p,'decode', 1, @isnumeric); % 1: decode during training
  addOptional(p,'minLenRatio', 0.5, @isnumeric);
  addOptional(p,'maxLenRatio', 1.5, @isnumeric);
  
  % debugging
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need input arguments as toy data is automatically generated.
  addOptional(p,'dataType', 'single', @ischar); % Note: use double precision for grad check
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'debug', 0, @isnumeric); % 0: no debug, 1: debug
  addOptional(p,'assert', 0, @isnumeric); % 0: no sanity check, 1: yes
  addOptional(p,'seed', 0, @isnumeric); % 0: seed based on current clock time, else use the specified seed
  
  %% attention! %%
  % attnFunc=0: no attention.
  %          1: global attention
  %          2: local attention + monotonic alignments
  %          4: local attention  + regression for absolute pos (multiplied distWeights)
  addOptional(p,'attnFunc', 0, @isnumeric);
  % attnOpt: decide how we generate the alignment weights:
  %          1: src compare, dot product, a_t = softmax(H_src * h_t)
  %          2: src compare, general dot product, a_t = softmax(H_src * W_a * h_t)
  %          3: src compare, general dot product, a_t = softmax(v_a*f(W_a * [H_src; h_t])
  addOptional(p,'attnOpt', 0, @isnumeric);
  addOptional(p,'posWin', 10, @isnumeric); % relative window, used for attnFunc~=1
  
  %% system options
  addOptional(p,'onlyCPU', 0, @isnumeric); % 1: avoid using GPUs
  addOptional(p,'gpuDevice', 0, @isnumeric); % choose the gpuDevice to use: 0 -- no GPU.

  p.KeepUnmatched = true;
  parse(p,trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,varargin{:})
  params = p.Results;

  %% Setup params
  params.chunkSize = params.batchSize*100;
  params.baseIndex = 0; %  the minimum value in all sequences of integers (often 0). Required to convert them to 1-indexed for Matlab.
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
  params.forceDecoder = 0;
  
  % params assertions
  if params.attnFunc==4
    assert(params.isReverse==1);
  end
  
  % rand seed
  if params.isGradCheck || params.isProfile || params.seed
    s = RandStream('mt19937ar','Seed',params.seed);
  else
    s = RandStream('mt19937ar','Seed','shuffle');
  end
  RandStream.setGlobalStream(s);
  
  % check GPUs
  params.isGPU = 0;
  if params.gpuDevice
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf(2, '# %d GPUs exist. So, we will use GPUs.\n', n);
      params.isGPU = 1;
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
    params.maxSentLen = 7;
    params.posWin = 1;
  end
  
  %% attention
  params.align = (params.attnFunc>0); % for the decoder 
  if params.attnFunc>0
    assert(params.attnOpt>0);
    params.attnGlobal = (params.attnFunc==1);
    if params.attnGlobal == 0 % local attention, predictive alignemtns       
      params.distSigma = params.posWin/2.0;
    end
  end
  
  %% log
  assert(strcmp(outDir, '')==0);
  if ~exist(outDir, 'dir')
    mkdir(outDir);
  end
  params.logId = fopen([outDir '/log'], 'a');
  
  %% Load vocabs
  % char
  if params.charShortList > 0
    params.srcCharVocabFile = [params.charPrefix '.' params.srcLang '.char.vocab'];
    params.srcCharMapFile = [params.charPrefix '.' params.srcLang '.char.map'];
    params.tgtCharVocabFile = [params.charPrefix '.' params.tgtLang '.char.vocab'];
    params.tgtCharMapFile = [params.charPrefix '.' params.tgtLang '.char.map'];
  end
  [params] = prepareVocabs(params);

  
  %% Init / Load Model Parameters
  params.modelFile = [outDir '/model.mat']; % store those with the best valid perplexity
  params.modelRecentFile = [outDir '/modelRecent.mat'];
  [model, params] = initLoadModel(params);
  % for backward compatibility  
  [params] = backwardCompatible(params, {'epochIter', 'saveHDF'});

  % print
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
  [validData] = loadPrepareData(params, params.validPrefix, params.srcVocab, params.tgtVocab);
  [testData] = loadPrepareData(params, params.testPrefix, params.srcVocab, params.tgtVocab);
  
  % train
  params.tgtTrainFile = sprintf('%s.%s', params.trainPrefix, params.tgtLang);
  if params.isBi
    if params.isReverse
      params.srcTrainFile = sprintf('%s.%s.reversed', params.trainPrefix, params.srcLang);
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
    printSent(2, srcTrainSents{1}, params.srcVocab, '  src 1:');
    printSent(2, srcTrainSents{end}, params.srcVocab, '  src end:');
  end
  printSent(2, tgtTrainSents{1}, params.tgtVocab, '  tgt:');
  printSent(2, tgtTrainSents{end}, params.tgtVocab, '  tgt end:');
  printTrainBatch(trainBatches{1}, params);
  
  
  %%%%%%%%%%%%%%
  %% Training %%
  %%%%%%%%%%%%%%
  params.totalLog = 0;
  params.evalFreq = params.logFreq*10;

  % profile
  if params.isProfile
    profile on
  end
  
  startTime = clock;
  fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  fprintf(params.logId, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  printCell(2, params.varsDenseUpdate, '# varsDenseUpdate: ');
  
  isRun = 1;
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
        if params.isClip==1
          fprintf(2, 'epoch=%d, iter=%d, nan/inf even with grad clipping ... No hope!\n', params.epoch, params.iter);
          isRun = 0;
          break;
        else
          fprintf(2, 'epoch=%d, iter=%d, nan/inf. Let us, restart and do grad clipping!\n', params.epoch, params.iter);
          [model] = initLoadModel(params);
          params.isClip = 1;
          continue;
        end
      end
      
      %% grad clipping      
      [gradNorm, ~] = computeGradNorm(grad, params.batchSize, params.varsDenseUpdate); % historical reason: we exclude W_emb % indNorms
      scale = 1.0/params.batchSize; % grad is divided by batchSize
      if gradNorm > params.maxGradNorm
        scale = scale*params.maxGradNorm/gradNorm;
      end
      scaleLr = params.lr*scale;
      
      %% update parameters
      for ii=1:length(params.varsDenseUpdate)
        field = params.varsDenseUpdate{ii};
        if iscell(model.(field))
          for jj=1:length(model.(field)) % cell, like W_src, W_tgt
            model.(field){jj} = model.(field){jj} - scaleLr*grad.(field){jj};
          end
        else
          model.(field) = model.(field) - scaleLr*grad.(field);
        end
      end
      % update W_emb sparsely
      if params.isBi
        model.W_emb_src(:, grad.indices_src) = model.W_emb_src(:, grad.indices_src) - scaleLr*grad.W_emb_src;  
      end
      % decoder
      model.W_emb_tgt(:, grad.indices_tgt) = model.W_emb_tgt(:, grad.indices_tgt) - scaleLr*grad.W_emb_tgt;
      
      % char
      if params.charShortList
        if params.isBi
          model.W_emb_src_char(:, grad.indices_src_char) = model.W_emb_src_char(:, grad.indices_src_char) - scaleLr*grad.W_emb_src_char;  
        end
        model.W_emb_tgt_char(:, grad.indices_tgt_char) = model.W_emb_tgt_char(:, grad.indices_tgt_char) - scaleLr*grad.W_emb_tgt_char;
      end

      %% logging, eval, save, decode, fine-tuning, etc.
      [params, startTime] = postTrainIter(model, costs, gradNorm, trainData, validData, testData, params, startTime, srcTrainSents, tgtTrainSents);
    end % end for batchId

    %% read more data
    [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params);
    
    %% end of an epoch
    if numTrainSents == 0 
      [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents, startTime, params] = postTrainEpoch(model, validData, testData, startTime, params);
      
      if params.epoch > params.numEpoches
        break; 
      end
    end
  end % end for while(1)
  
  fclose(params.logId);
end

function [params] = initTrainParams(params)
  params.lr = params.learningRate;
  params.epoch = 1;
  params.bestCostValid = 1e5;
  params.testPerplexity = 1e5;
  params.curTestPerpWord = 1e5;
  params.startIter = 0;
  params.iter = 0;  % number of batches we have processed
  params.epochBatchCount = 0;
  params.finetuneCount = 0;
  params.trainCounts = initCosts();
  params.trainCosts = initCosts();
end

%% Init model parameters
function [model] = initLSTM(params)
  fprintf(2, '# Init LSTM parameters using dataType=%s, initRange=%g\n', params.dataType, params.initRange);
  
  % W_src
  if params.isBi
    model.W_src = cell(params.numLayers, 1);    
    for ll=1:params.numLayers
      model.W_src{ll} = initMatrixRange(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
  end
  
  % W_tgt
  model.W_tgt = cell(params.numLayers, 1);
  for ll=1:params.numLayers
    model.W_tgt{ll} = initMatrixRange(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
  end
  if params.feedInput % feed in src hidden states to the tgt
    model.W_tgt{1} = initMatrixRange(params.initRange, [4*params.lstmSize, 3*params.lstmSize], params.isGPU, params.dataType);
  end
  
  %% NOTE: convention here, parameters under model struct that starts with W_emb are updated sparsely.
  % W_emb
  if params.charShortList > 0 % hybrid
    % src
    if params.isBi
      % word
      model.W_emb_src = initMatrixRange(params.initRange, [params.lstmSize, params.charShortList], params.isGPU, params.dataType);
      % char
      model.W_src_char = cell(params.charNumLayers, 1);    
      for ll=1:params.charNumLayers
        model.W_src_char{ll} = initMatrixRange(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
      end
      model.W_emb_src_char = initMatrixRange(params.initRange, [params.lstmSize, params.srcCharVocabSize], params.isGPU, params.dataType);
    end
    
    % tgt
    % word
    model.W_emb_tgt = initMatrixRange(params.initRange, [params.lstmSize, params.charShortList], params.isGPU, params.dataType);
    
    % char
    model.W_tgt_char = cell(params.charNumLayers, 1);    
    for ll=1:params.charNumLayers
      model.W_tgt_char{ll} = initMatrixRange(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
    model.W_emb_tgt_char = initMatrixRange(params.initRange, [params.lstmSize, params.tgtCharVocabSize], params.isGPU, params.dataType);
  else % word
    if params.isBi
      model.W_emb_src = initMatrixRange(params.initRange, [params.lstmSize, params.srcVocabSize], params.isGPU, params.dataType);
    end
    model.W_emb_tgt = initMatrixRange(params.initRange, [params.lstmSize, params.tgtVocabSize], params.isGPU, params.dataType);
  end
  
  
  %% h_t -> softmax input
  if params.attnFunc>0 % attention mechanism
    % local attention, predict positions
    if params.attnGlobal == 0
      % transform h_t into h_pos = f(W_pos*h_t)
      model.W_pos = initMatrixRange(params.initRange, [params.softmaxSize, params.lstmSize], params.isGPU, params.dataType);
      
      % regression, scale=sigmoid(v_pos*h_pos)
      model.v_pos = initMatrixRange(params.initRange, [1, params.softmaxSize], params.isGPU, params.dataType);
    end
    
    % predict alignment weights
    % content-based alignments
    if params.attnOpt==1 % dot product, nothing to do here, softmax(H_src*h_t))
    elseif params.attnOpt==2 % general dot product: softmax(H_src*W_a*h_t))
      model.W_a = initMatrixRange(params.initRange, [params.lstmSize, params.lstmSize], params.isGPU, params.dataType);
    elseif params.attnOpt==3 % similar to Bengio's style, plus: softmax(v_a*f(H_src + W_a*h_t))
      model.W_a = initMatrixRange(params.initRange, [params.lstmSize, params.lstmSize], params.isGPU, params.dataType);
      model.v_a = initMatrixRange(params.initRange, [1, params.lstmSize], params.isGPU, params.dataType);
    end
    
    % attn_t = H_src * a_t % h_attn_t = f(W_h * [attn_t; h_t])
    model.W_h = initMatrixRange(params.initRange, [params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
  end
  
  %% softmax input -> predictions
  if params.charShortList % hybrid, shortList + <rare>
    model.W_soft = initMatrixRange(params.initRange, [params.charShortList + 1, params.softmaxSize], params.isGPU, params.dataType);
  else
    model.W_soft = initMatrixRange(params.initRange, [params.tgtVocabSize, params.softmaxSize], params.isGPU, params.dataType);
  end
end

%% Things to do after each training iteration %%
function [params, startTime] = postTrainIter(model, costs, gradNorm, trainData, validData, testData, params, startTime, srcTrainSents, tgtTrainSents)
  %% log info
  params.trainCounts = updateCounts(params.trainCounts, trainData);
  params.totalLog = params.totalLog + trainData.numWords; % to compute speed
  
  [params.trainCosts] = updateCosts(params.trainCosts, costs);
  
  if mod(params.iter, params.logFreq) == 0
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

    endTime = clock;
    timeElapsed = etime(endTime, startTime);
    params.speed = params.totalLog*0.001/timeElapsed;
    
    [params.scaleTrainCosts] = scaleCosts(params.trainCosts, params.trainCounts);
    logStr = sprintf('%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s', params.epoch, params.iter, params.speed, params.lr, ...
        params.scaleTrainCosts.total, gradNorm, datestr(now));
    
    fprintf(2, '%s\n', logStr);
    fprintf(params.logId, '%s\n', logStr);
  
    % reset
    params.totalLog = 0;
    startTime = clock;
  end

  %% eval
  if mod(params.iter, params.evalFreq) == 0    
    % eval, save, and decode
    [params] = evalSaveDecode(model, validData, testData, params, srcTrainSents, tgtTrainSents);

    startTime = clock;
  end

  % finetuning
  if params.epochIter > 0 % file large, use epochIter
    if (params.iter / params.epochIter) >= params.finetuneEpoch && mod(params.iter, floor(params.epochIter*params.epochFraction))==0 
      params = finetune(params);
    end
  elseif params.epoch > params.finetuneEpoch && mod(params.iter, params.finetuneCount)==0
    params = finetune(params);
  end

  if params.epoch==1
    params.epochBatchCount = params.epochBatchCount + 1;
  end
end

function [params] = finetune(params)
  fprintf(2, '# Finetuning at epoch %d, iter %d, %g -> %g\n', params.epoch, params.iter, params.lr, params.lr*params.finetuneRate);
  fprintf(params.logId, '# Finetuning at epoch %d, iter %d, %g -> %g\n', params.epoch, params.iter, params.lr, params.lr*params.finetuneRate);
  params.lr = params.lr*params.finetuneRate;
end

%% Things to do at the end of each training epoch %%
function [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents, startTime, params] = postTrainEpoch(model, validData, testData, startTime, params)
  % seek to the beginning
  if params.isBi
    fseek(params.srcTrainId, 0, 'bof');
  end
  fseek(params.tgtTrainId, 0, 'bof');

  % for cases where logFreq > number of batches in a n epoch
  if ~isfield(params, 'costTrain') || ~isfield(params, 'speed')
    endTime = clock;
    timeElapsed = etime(endTime, startTime);
    params.costTrain = params.trainCosts.total/params.trainCounts.total;
    params.speed = params.totalLog*0.001/timeElapsed;
    params.totalLog = 0;
    startTime = clock;
  end
  
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
  end
end

function [params] = evalSaveDecode(model, validData, testData, params, srcTrainSents, tgtTrainSents)
  % eval
  [params] = evalValidTest(model, validData, testData, params);

  % save
  fprintf(2, '  save model cur test perplexity %.2f to %s\n', params.curTestPerpWord, params.modelRecentFile);
  fprintf(params.logId, '  save model cur test perplexity %.2f to %s\n', params.curTestPerpWord, params.modelRecentFile);
  save(params.modelRecentFile, 'model', 'params');
  if params.saveHDF
    saveHDF5([params.modelRecentFile '.h5'], model, params);
  end
  
  % decode
  if params.isBi && params.decode==1
    validId = randi(validData.numSents);
    testId = randi(testData.numSents);
    decodeSent(srcTrainSents(1), tgtTrainSents(1), model, params);
    decodeSent(validData.srcSents(validId), validData.tgtSents(validId), model, params);
    decodeSent(testData.srcSents(testId), testData.tgtSents(testId), model, params);
  end
end

function decodeSent(srcSent, tgtSent, model, params)
  params.preeosId = -1;
  [decodeData] = prepareData(srcSent, tgtSent, 1, params);
  decodeData.startId = 1;
  [candidates, candScores, alignInfo, otherInfo] = lstmDecoder(model, decodeData, params);
  printDecodeResults(decodeData, candidates, candScores, alignInfo, otherInfo, params, 0);
end

function [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params)
  if params.isBi
    [srcTrainSents, ~, srcTrainLens] = loadBatchData(params.srcTrainId, params.baseIndex, params.chunkSize);
  else
    srcTrainSents = {};
  end
  [tgtTrainSents, numTrainSents, tgtTrainLens] = loadBatchData(params.tgtTrainId, params.baseIndex, params.chunkSize);
  
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

function [model, params, oldParams, loaded] = loadModel(modelFile, params, isFreshTrain)
% isFreshTrain=1: we get the parameters but used the init learning rate,
% epoch, iter, etc.
  loaded = 0;
  model = [];
  oldParams = [];
  if exist(modelFile, 'file')
    fprintf(2, '# Model file %s exists. Try loading ...\n', modelFile);
    fprintf(params.logId, '# Model file %s exists. Try loading ...\n', modelFile);
    try
      savedData = load(modelFile);
      loaded = 1;
    catch ME
      model = [];
      fprintf(2, '! Exception: identifier=%s, name=%s\n', ME.identifier, ME.message);
      return;
    end
  else
    fprintf(2, '! File %s doesnot exist\n', modelFile);
    return;
  end

  % params
  oldParams = savedData.params;
  if isFreshTrain
    [params] = initTrainParams(params);
  else
    params.lr = oldParams.lr;
    params.epoch = oldParams.epoch;
    params.epochBatchCount = oldParams.epochBatchCount;
    params.bestCostValid = oldParams.bestCostValid;
    params.testPerplexity = oldParams.testPerplexity;
    params.trainCounts = oldParams.trainCounts;
    params.trainCosts = oldParams.trainCosts;
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
  end
  
  % model
  model = savedData.model;
  model
  clear savedData;

  fprintf(2, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g, model: %s\n', params.lr, params.epoch, params.startIter, params.bestCostValid, params.testPerplexity, wInfo(model));
  fprintf(params.logId, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g, model: %s\n', params.lr, params.epoch, params.startIter, params.bestCostValid, params.testPerplexity, wInfo(model));
end

function [model, params] = initLoadModel(params)
  % softmaxSize
  params.softmaxSize = params.lstmSize;
  
  % a model exists, resume training
  loaded = 0;
  if params.isGradCheck==0
    if (strcmp(params.loadModel, '')==0 && exist(params.loadModel, 'file')) % load from a specified model
      [model, params, ~, loaded] = loadModel(params.loadModel, params, 1);
      if loaded == 0
        error('Failed to load model %s\n', params.loadModel);
      end
    elseif params.isResume && (exist(params.modelRecentFile, 'file') || exist(params.modelFile, 'file')) % resume training
      [model, params, ~, loaded] = loadModel(params.modelRecentFile, params, 0);
      if loaded == 0 && exist(params.modelFile, 'file')
        [model, params, ~, loaded] = loadModel(params.modelFile, params, 0);
      end

      if loaded==0
        error('! Failed to load model files\n');
      end
    end
  end
  
  if loaded == 0 % start from scratch
    [model] = initLSTM(params);
    [params] = initTrainParams(params);
  end

  % compute model size
  params.modelSize = modelSizes(model);
  
  params = setupVars(model, params);
end

function [params] = setupVars(model, params)
  params.vars = fields(model);
  
  % exclude W_emb and W_soft_inclass (for class-based softmax). These are
  % those matrices which we will update sparsely
  params.varsDenseUpdate = {};
  for ii=1:length(params.vars)
    if strncmp(params.vars{ii}, 'W_emb', 4)==0
      params.varsDenseUpdate{end+1} = params.vars{ii};
    end
  end
end

function [data] = loadPrepareData(params, prefix, srcVocab, tgtVocab)
  [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab);
  [data] = prepareData(srcSents, tgtSents, 1, params);
  fprintf(2, '  numSents=%d, numWords=%d\n', numSents, data.numWords);
  data.numSents = numSents;
  data.srcSents = srcSents;
  data.tgtSents = tgtSents;
end
