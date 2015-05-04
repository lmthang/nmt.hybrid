%% Train Long-Short Term Memory (LSTM).
% Thang Luong @ 2014, 2015, <lmthang@stanford.edu>
% With contributions from:
%   Hieu Pham: decoder.
%
% Options:
%   srcLang, tgtLang: languages, e.g. en, de. (leave srcLang empty for monolingual models)
%   srcVocabFile, tgtVocabFile: for verifying that we correctly map indices to words (leave srcVocabFile empty for monolingual models)
%   trainPrefix, validPrefix, testPrefix: we will use trainPrefix.srcLang,
%     train Prefix.tgtLang files for training, and similarly for validating
%     and testing. These data files contain sequences of integers one per line.
%   outDir: output directory.
%   baseIndex: the minimum value in all sequences of integers (often 0). Required to convert them to 1-indexed for Matlab.
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
  addOptional(p,'dataType', 'single', @ischar); % Note: use double precision for grad check
  addOptional(p,'maxSentLen', -1, @isnumeric); % mostly apply to src, used in attention-based models. Default: 50 + 1 (eos). For positional models, we use maxSentLen = (maxSentLen-1)*2+1 for the tgt side
  addOptional(p,'sortBatch', 0, @isnumeric); % 1: each time we read in 100 batches, we sort sentences by length.
  addOptional(p,'shuffle', 0, @isnumeric); % 1: shuffle training batches
  addOptional(p,'decode', 1, @isnumeric); % 1: decode during training

  %% decoding
  addOptional(p,'minLenRatio', 0.5, @isnumeric);
  addOptional(p,'maxLenRatio', 1.5, @isnumeric);
  addOptional(p,'depParse', 0, @isnumeric); % 1: indicate that we are doing dependency parsing
  
  %% debugging options
  addOptional(p,'isGradCheck', 0, @isnumeric); % set 1 to check the gradient, no need input arguments as toy data is automatically generated.
  addOptional(p,'isProfile', 0, @isnumeric);
  addOptional(p,'debug', 0, @isnumeric); % 0: no debug, 1: debug
  addOptional(p,'assert', 0, @isnumeric); % 0: no sanity check, 1: yes
  addOptional(p,'seed', 0, @isnumeric); % 0: seed based on current clock time, else use the specified seed
  
  %% advanced (working) features
  addOptional(p,'dropout', 1, @isnumeric); % dropout prob: 1 no dropout, <1: dropout
  addOptional(p,'softmaxDim', 0, @isnumeric); % softmaxDim>0 convert hidden state into an intermediate representation of size softmaxDim before going through the softmax
  % attnFunc=0: no attention.
  %          >0: a_t = softmax(W_a * [tgt_h_t; srcLens])  
  %           1: absolute positions
  %           2: relative positions
  %           3: absolute positions + feed to input (start compute attn from srcMaxLen - 1)
  %           4: relative positions + feed to input (start compute attn from srcMaxLen - 1)
  addOptional(p,'attnFunc', 0, @isnumeric);
  addOptional(p,'attnSize', 0, @isnumeric); % dim of the vector used to input to the final softmax, if 0, use lstmSize
  addOptional(p,'posWin', 20, @isnumeric); % relative window, also used in positional models 

  %% research options  
  % positional models: predict pos, then word, use a separate softmax for pos
  % 0: separately print out pos/word perplexities
  % 1: predict pos/word with a separate softmax W_softPos
  % 2: like 1 + feed in src hidden states to compute tgt hidden states
  % 3: like 1 + feed in src hidden states to compute softmax
  addOptional(p,'posModel', -1, @isnumeric);  
  addOptional(p,'lstmOpt', 0, @isnumeric); % lstmOpt=0: basic model, 1: no tanh for c_t.
  addOptional(p,'sameLength', 0, @isnumeric); % sameLength=1: output and input are of the same length, so let's feed the src hidden states into the tgt!
  
  addOptional(p,'monoFile', '', @ischar); % to bootstrap the decoder with a monolingual model
  addOptional(p,'decodeUpdateEpoch', 1, @isnumeric); % when to start updating the pretrained decoder epoch>=monoUpdateEpoch (1 means start updating at the very beginning).
  addOptional(p,'decodeUpdateOpt', 0, @isnumeric); % 0: everything is fixed, 1: only update LSTM params, 2: only update softmax params (these options only activate when decodeUpdateEpoch>1).
  
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
 
  % decode params
  params.beamSize = 12;
  params.stackSize = 100;
  params.unkPenalty = 0;
  params.lengthReward = 0;
  
  % maxSentLen
  if params.maxSentLen==-1
    params.maxSentLen = 51;
  end
  
  % params assertions
  if params.posModel>=0
    assert(params.isBi==1);
    assert(params.attnFunc==0);
  end
  if params.attnFunc>0 || params.sameLength==1
    assert(params.isReverse==1);
  end
  if params.softmaxDim>0
    assert(params.attnFunc==0 & params.posModel==-1, '! Assert failed: softmaxDim %d > 0, so attnFunc %d and posModel %d have to be -1.\n', ...
      params.softmaxDim, params.attnFunc, params.posModel);
  end
  
  % rand seed
  if params.isGradCheck || params.isProfile || params.seed
    s = RandStream('mt19937ar','Seed',params.seed)
  else
    s = RandStream('mt19937ar','Seed','shuffle')
  end
  RandStream.setGlobalStream(s);
  
  % check GPUs
  params.isGPU = 0;
  if ismac==0 && params.onlyCPU==0
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
  
  %% set more params
  % attentional/positional models
  if params.attnFunc>0 || params.posModel>0
    if params.attnSize==0
      params.attnSize = params.lstmSize;
    end
    
    if params.attnFunc==2 || params.attnFunc==4 % relative positions
      params.numAttnPositions = 2*params.posWin + 1;
    elseif params.attnFunc==1 || params.attnFunc==3 % absolute positions
      params.numAttnPositions = params.maxSentLen-1;
    end
  end
  
  assert(strcmp(outDir, '')==0);
  if ~exist(outDir, 'dir')
    mkdir(outDir);
  end
  params.logId = fopen([outDir '/log'], 'a');
  
  %% Load vocabs
  [params] = loadBiVocabs(params);
  % dependency parsing
  if params.depParse 
    params.depRootId = find(strcmp(params.tgtVocab, 'R(root)')==1,1);
    params.depShiftId = find(strcmp(params.tgtVocab, 'S')==1,1);
    fprintf(2, '# Dependency parsing, rootId for %s=%d, shiftId for %s=%d\n', params.tgtVocab{params.depRootId}, params.depRootId, ...
      params.tgtVocab{params.depShiftId}, params.depShiftId);
  end
  
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
  [validData] = loadPrepareData(params, params.validPrefix, params.srcVocab, params.tgtVocab);
  [testData] = loadPrepareData(params, params.testPrefix, params.srcVocab, params.tgtVocab);
  
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
    printSent(2, srcTrainSents{1}, params.srcVocab, '  src 1:');
    printSent(2, srcTrainSents{end}, params.srcVocab, '  src end:');
  end
  printSent(2, tgtTrainSents{1}, params.tgtVocab, '  tgt:');
  printSent(2, tgtTrainSents{end}, params.tgtVocab, '  tgt end:');
  printTrainBatch(trainBatches{1}, params);
  
  
  %%%%%%%%%%%%%%
  %% Training %%
  %%%%%%%%%%%%%%
  trainCost.total = 0; trainWords.total = 0;
  trainWords.totalLog = 0;
  if params.posModel>=0 % positional model
    trainCost.pos = 0;
    trainCost.word = 0;
  end
  params.evalFreq = params.logFreq*10;

  % profile
  if params.isProfile
    profile on
  end
  
  startTime = clock;
  fprintf(2, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  fprintf(params.logId, '# Epoch %d, lr=%g, %s\n', params.epoch, params.lr, datestr(now));
  
  % not update decoder
  if params.epoch<params.decodeUpdateEpoch
    % fix decoder LSTM params
    if params.decodeUpdateOpt==0 || params.decodeUpdateOpt==2 
      index = find(strcmp('W_tgt', params.varsDenseUpdate)==1, 1);
      params.varsDenseUpdate(index) = [];
    end
    % fix decoder softmax params
    if params.decodeUpdateOpt==0 || params.decodeUpdateOpt==1 
      index = find(strcmp('W_soft', params.varsDenseUpdate)==1, 1);
      params.varsDenseUpdate(index) = [];
    end
  end
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
      % update W_emb separately
      if params.isBi
        model.W_emb_src(:, grad.indices_src) = model.W_emb_src(:, grad.indices_src) - scaleLr*grad.W_emb_src;  
      end
      % update the decoder
      if params.epoch>=params.decodeUpdateEpoch
        model.W_emb_tgt(:, grad.indices_tgt) = model.W_emb_tgt(:, grad.indices_tgt) - scaleLr*grad.W_emb_tgt;
      end
      
      %% logging, eval, save, decode, fine-tuning, etc.
      [trainWords, trainCost, params, startTime] = postTrainIter(model, costs, gradNorm, trainData, validData, testData, trainWords, trainCost, params, startTime, srcTrainSents, tgtTrainSents);
    end % end for batchId

    %% read more data
    [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents] = loadTrainBatches(params);
    
    %% end of an epoch
    if numTrainSents == 0 
      [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents, startTime, params] = postTrainEpoch(model, validData, testData, startTime, trainWords, trainCost, params);
      
      if params.epoch > params.numEpoches
        break; 
      end
    end
  end % end for while(1)
  
  fclose(params.logId);
end

%% Init model parameters
function [model] = initLSTM(params)
  fprintf(2, '# Init LSTM parameters using dataType=%s, initRange=%g\n', params.dataType, params.initRange);
  
  % W_src
  if params.isBi
    model.W_src = cell(params.numLayers, 1);    
    for ll=1:params.numLayers
      model.W_src{ll} = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
  end
  
  % W_tgt
  model.W_tgt = cell(params.numLayers, 1);
  for ll=1:params.numLayers
    model.W_tgt{ll} = randomMatrix(params.initRange, [4*params.lstmSize, 2*params.lstmSize], params.isGPU, params.dataType);
  end
  if params.sameLength==1 || params.attnFunc==3 || params.attnFunc==4 % feed in src hidden states to the tgt
    model.W_tgt{1} = randomMatrix(params.initRange, [4*params.lstmSize, 3*params.lstmSize], params.isGPU, params.dataType);
  end
  
  % W_emb
  if params.isBi
    model.W_emb_src = randomMatrix(params.initRange, [params.lstmSize, params.srcVocabSize], params.isGPU, params.dataType);
    model.W_emb_src(:, params.srcZero) = zeros(params.lstmSize, 1);
  end
  model.W_emb_tgt = randomMatrix(params.initRange, [params.lstmSize, params.tgtVocabSize], params.isGPU, params.dataType);
  model.W_emb_tgt(:, params.tgtEos) = zeros(params.lstmSize, 1);
    
  %% h_t -> softmax input
  % attention mechanism
  if params.attnFunc>0 
    model.W_a = randomMatrix(params.initRange, [params.numAttnPositions, params.lstmSize], params.isGPU, params.dataType);
    
    if params.attnFunc==1 || params.attnFunc==2 % attn_t = H_src * a_t % h_attn_t = f(W_ah * [attn_t; h_t])
      model.W_ah = randomMatrix(params.initRange, [params.attnSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
  % compress softmax
  elseif params.softmaxDim>0 
    model.W_h = randomMatrix(params.initRange, [params.softmaxDim, params.lstmSize], params.isGPU, params.dataType);
  % positional models
  elseif params.posModel>0 
    if params.posModel==2 % W_tgt [x_t; h_t] + W_tgt_pos*s_t, where s_t is the source hidden state and is used to compute the tgt hidden states
      model.W_tgt_pos = randomMatrix(params.initRange, [4*params.lstmSize, params.lstmSize], params.isGPU, params.dataType);
    elseif params.posModel==3 % h_pos_t = f(W_h * [src_pos_t; h_t])
      model.W_h = randomMatrix(params.initRange, [params.attnSize, 2*params.lstmSize], params.isGPU, params.dataType);
    end
    
    model.W_softPos = randomMatrix(params.initRange, [params.posVocabSize, params.softmaxSize], params.isGPU, params.dataType);
  end
  
  %% softmax input -> predictions
  % W_soft
  model.W_soft = randomMatrix(params.initRange, [params.tgtVocabSize, params.softmaxSize], params.isGPU, params.dataType);
end

%% Things to do after each training iteration %%
function [trainWords, trainCost, params, startTime] = postTrainIter(model, costs, gradNorm, trainData, validData, testData, trainWords, trainCost, params, startTime, srcTrainSents, tgtTrainSents)
  %% log info
  trainWords.total = trainWords.total + trainData.numWords;
  trainCost.total = trainCost.total + costs.total;
  trainWords.totalLog = trainWords.totalLog + trainData.numWords; % to compute speed
  if params.posModel>=0 % positional model
    trainCost.pos = trainCost.pos + costs.pos;
    trainCost.word = trainCost.word + costs.word;
  end
  if mod(params.iter, params.logFreq) == 0
    endTime = clock;
    timeElapsed = etime(endTime, startTime);
    params.costTrain = trainCost.total/trainWords.total;
    params.speed = trainWords.totalLog*0.001/timeElapsed;
    if params.posModel>=0 % positional model
      params.speed = params.speed/2;
      params.costTrainPos = trainCost.pos*2/trainWords.total;
      params.costTrainWord = trainCost.word*2/trainWords.total;
      fprintf(2, '%d, %d, %.2fK, %g, %.2f (%.2f, %.2f), gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, params.costTrainPos, params.costTrainWord, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
      fprintf(params.logId, '%d, %d, %.2fK, %g, %.2f (%.2f, %.2f), gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, params.costTrainPos, params.costTrainWord, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
    else
      fprintf(2, '%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
      fprintf(params.logId, '%d, %d, %.2fK, %g, %.2f, gN=%.2f, %s\n', params.epoch, params.iter, params.speed, params.lr, params.costTrain, gradNorm, datestr(now)); % , wInfo(indNorms, 1)
    end

    % reset
    trainWords.totalLog = 0;
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
end

%% Things to do at the end of each training epoch %%
function [trainBatches, numTrainSents, numBatches, srcTrainSents, tgtTrainSents, startTime, params] = postTrainEpoch(model, validData, testData, startTime, trainWords, trainCost, params)
  % seek to the beginning
  if params.isBi
    fseek(params.srcTrainId, 0, 'bof');
  end
  fseek(params.tgtTrainId, 0, 'bof');

  % for cases where logFreq > number of batches in a n epoch
  if ~isfield(params, 'costTrain') || ~isfield(params, 'speed')
    endTime = clock;
    timeElapsed = etime(endTime, startTime);
    params.costTrain = trainCost.total/trainWords.total;
    params.speed = trainWords.totalLog*0.001/timeElapsed;
    trainWords.totalLog = 0;
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
  
  % now update decoder
  if params.decodeUpdateEpoch>1 && params.epoch==params.decodeUpdateEpoch
    % LSTM params
    if params.decodeUpdateOpt==0 || params.decodeUpdateOpt==2 
      params.varsDenseUpdate{end+1} = 'W_tgt';
    end
    % fix decoder softmax params
    if params.decodeUpdateOpt==0 || params.decodeUpdateOpt==1 
      params.varsDenseUpdate{end+1} = 'W_soft';
    end
    printCell(2, params.varsDenseUpdate, '# update varsDenseUpdate: ');
  end
  

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
  fprintf(2, '  save model cur test perplexity %.2f to %s\n', params.curTestPerplexity, params.modelRecentFile);
  fprintf(params.logId, '  save model cur test perplexity %.2f to %s\n', params.curTestPerplexity, params.modelRecentFile);
  save(params.modelRecentFile, 'model', 'params');

  % decode
  if params.isBi && params.posModel<=0 && params.decode==1
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
  [candidates, candScores] = lstmDecoder(model, decodeData, params);
  printDecodeResults(decodeData, candidates, candScores, params, 0);
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

function [model, params, oldParams, loaded] = loadModel(modelFile, params)
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
  model
  clear savedData;

  fprintf(2, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g, model: %s\n', params.lr, params.epoch, params.startIter, params.bestCostValid, params.testPerplexity, wInfo(model));
  fprintf(params.logId, '  loaded! lr=%g, epoch=%d, iter=%d, bestCostValid=%g, testPerplexity=%g, model: %s\n', params.lr, params.epoch, params.startIter, params.bestCostValid, params.testPerplexity, wInfo(model));
end

function [model, params] = initLoadModel(params)
  % softmaxSize
  if params.attnFunc>0 || params.posModel>0 % attention/positional mechanism    
    params.softmaxSize = params.attnSize;
  elseif params.softmaxDim>0 % compress softmax
    params.softmaxSize = params.softmaxDim;
  else % normal
    params.softmaxSize = params.lstmSize;
  end
  
  % a model exists, resume training
  if params.isGradCheck==0 && params.isResume && (exist(params.modelRecentFile, 'file') || exist(params.modelFile, 'file'))
    [model, params, ~, loaded] = loadModel(params.modelRecentFile, params);
    if loaded == 0 && exist(params.modelFile, 'file')
      [model, params, ~, loaded] = loadModel(params.modelFile, params);
    end
    
    if loaded==0
      error('! Failed to load model files\n');
    end
  else % start from scratch
    [model] = initLSTM(params);
    params.lr = params.learningRate;
    params.epoch = 1;
    params.bestCostValid = 1e5;
    params.testPerplexity = 1e5;
    params.curTestPerplexity = 1e5;
    params.startIter = 0;
    params.iter = 0;  % number of batches we have processed
    params.epochBatchCount = 0;
    params.finetuneCount = 0;
    
    
    %% mono-bootstrap
    if strcmp(params.monoFile, '')==0
      [monoModel, ~, monoParams, loaded] = loadModel(params.monoFile, params);
      if loaded==0
        error('! Failed to load mono model %s\n', params.monoFile);
      end
      if isfield(monoParams, 'tgtVocab')
        monoParams.vocab = monoParams.tgtVocab;
      end
      if isfield(monoModel, 'W_emb_tgt')
        monoModel.W_emb = monoModel.W_emb_tgt;
      end
      
      fprintf('# Bootstrap from a mono model: W_tgt size=%s, vocab size=%d\n', mat2str(size(monoModel.W_tgt{1})), length(monoParams.vocab));
      fprintf('  W_tgt size=%s\n', mat2str(size(model.W_tgt{1})));
      fprintf('  tgt vocab size=%d\n', params.tgtVocabSize);
      
      % W_tgt
      assert(strcmp(mat2str(size(model.W_tgt{1})), mat2str(size(monoModel.W_tgt{1})))==1);
      model.W_tgt = monoModel.W_tgt;
      
      % vocab
      assert(params.tgtVocabSize==length(monoParams.vocab));
      flags = strcmp(params.tgtVocab, monoParams.vocab);
      matchCount = sum(flags);
      fprintf('  vocab match count=%d\n', matchCount);
      model.W_emb_tgt = monoModel.W_emb; % due to encoding problem, it's not quite correct at the moment to compare string like the below code. just hope that the two vocabs agree!
      %model.W_emb_tgt(:, flags) = monoModel.W_emb(:, flags);
      
      % handle mismatch vocab
      if matchCount < params.tgtVocabSize
       indices = find(flags==0);
       remainVocab = params.tgtVocab(indices);
       remainMonoVocab = params.tgtVocab(indices);
       for ii=1:length(indices)
         fprintf(2, '  mismatch %s \t %s\n', params.tgtVocab{indices(ii)}, monoParams.vocab{indices(ii)});
         %index = find(strcmp(remainVocab{ii}, remainMonoVocab), 1);
         %if isempty(index)
         %  fprintf(2, '  cannot init word %s\n', remainVocab{ii});
         %else
         %  model.W_emb_tgt(:, indices(ii)) = monoModel.W_emb(:, indices(index));
         %end
       end
       %assert(abs(matchCount-params.tgtVocabSize)<10);
      end
    end
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
    if strncmp(params.vars{ii}, 'W_emb', 4)==0 && strcmp(params.vars{ii}, 'W_soft_inclass')==0
      params.varsDenseUpdate{end+1} = params.vars{ii};
    end
  end
  
%   % setup softmax vars
%   if params.attnFunc==1 || params.attnFunc==2
%     softmaxVars = {'W_a', 'W_ah'};
%   elseif params.softmaxDim>0
%     softmaxVars = {'W_h'};
%   elseif params.posModel>0
%     if params.posModel==3
%       softmaxVars = {'W_h', 'W_softPos'};
%     else
%       softmaxVars = {'W_softPos'};
%     end
%   else
%     softmaxVars = {};
%   end
%   
%   softmaxVars{end+1} = 'W_soft';
%   params.softmaxVars = softmaxVars;
end

function [data] = loadPrepareData(params, prefix, srcVocab, tgtVocab)
  [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab);
  [data] = prepareData(srcSents, tgtSents, 1, params);
  fprintf(2, '  numSents=%d, numWords=%d\n', numSents, data.numWords);
  data.numSents = numSents;
  data.srcSents = srcSents;
  data.tgtSents = tgtSents;
end

%% class-based softmax %%
% addOptional(p,'numClasses', 0, @isnumeric); % >0: class-based softmax
% in initLSTM()
%   if params.numClasses == 0 % normal  
%   else % class-based
%     assert(mod(params.outVocabSize, params.numClasses) == 0, sprintf('outVocabSize (%d) must be divisible by numClasses (%d)', params.outVocabSize, params.numClasses));
%     
%     % W_soft_class: numClasses * softmaxSize
%     model.W_soft_class = randomMatrix(params.initRange, [params.numClasses, params.softmaxSize], params.isGPU, params.dataType);
%     
%     % W_soft_inclass: classSize * softmaxSize * numClasses
%     model.W_soft_inclass = randomMatrix(params.initRange, [params.classSize, params.softmaxSize, params.numClasses], params.isGPU, params.dataType);
%   end
% in initLoadModel()
%   % class-based softmax
%   if params.numClasses>0
%     params.classSize = params.outVocabSize / params.numClasses;
%   end
% in main loop update parameters
%       if params.numClasses>0
%         % update W_soft_inclass separately
%         model.W_soft_inclass(:, :, grad.classIndices) = model.W_soft_inclass(:, :, grad.classIndices) - scaleLr*grad.W_soft_inclass;
%       end
% in setupVars()
%   if params.numClasses == 0
%   else
%     softmaxVars{end+1} = 'W_soft_class';
%   end

%% Unused code %%
%       if params.separateEmb==1
%       else
%         model.W_emb(:, grad.indices) = model.W_emb(:, grad.indices) - scaleLr*grad.W_emb;
%       end

%   if params.separateEmb==1 % separate embeddings
%   else
%     model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], params.isGPU, params.dataType);
% 
%     % set parameters correspond to zero words
%     if params.isBi
%       model.W_emb(:, params.srcZero) = zeros(params.lstmSize, 1);
%     end
%     model.W_emb(:, params.tgtEos) = zeros(params.lstmSize, 1);
%   end


%     trainCost.total = 0;
%     if params.posModel>=0 % positional model
%       trainCost.pos = 0;
%       trainCost.word = 0;
%     end

%   addOptional(p,'embCPU', 0, @isnumeric); % 1: put W_emb on CPU even if GPUs exist
%   if params.embCPU == 1
%     fprintf(2, '# W_emb is explicitly put on CPU\n');
%     model.W_emb = randomMatrix(params.initRange, [params.lstmSize, params.inVocabSize], 0, 'double');
%   else  
%   end
%       if params.embCPU && params.isGPU
%         model.W_emb(:, grad.indices) = model.W_emb(:, grad.indices) - gather(scaleLr)*grad.W_emb;
%       else  
%       end
  
%   addOptional(p,'globalOpt', 0, @isnumeric); % globalOpt=0: no global model, 1: avg global model, 2: feedforward global model.
