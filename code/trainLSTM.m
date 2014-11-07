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
  p.KeepUnmatched = true;
  parse(p,trainPrefix,validPrefix,testPrefix,srcLang,tgtLang,srcVocabFile,tgtVocabFile,outDir,baseIndex,varargin{:})
  params = p.Results;
  
  % clip
  params.clipForward = 50; % clip c_t, h_t
  params.clipBackward = 1000; % clip dc, dh
  
  % check GPUs
  params.isGPU = 0;
  if ismac==0
    n = gpuDeviceCount;  
    params.srcLenLimit = 130;
    params.tgtLenLimit = 130;
    if n>0 % GPU exists
      fprintf('# %d GPUs exist. So, we will use GPUs.\n', n);
      params.isGPU = 1;
      for ii=1:n
        gpuDevice(ii)
      end
    end
  end
  
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

  % bilingual setting
  if strcmp(params.srcLang, '') == 0 % not empty, bilingual setting
    params.isBi = 1;
    fprintf(2, '## Bilingual setting\n');
  else
    params.isBi = 0;
    fprintf(2, '## Monolingual setting\n');
  end
  
  % grad check
  if params.isGradCheck
    %params.isGPU = 0;
    params.initRange = 0.1;
    params.lstmSize = 2;
    params.srcVocabSize = 3; 
    params.tgtVocabSize = 3;
    params.batchSize = 10;
  else
    % load vocab
    if params.isBi
      [srcVocab] = loadVocab(params.srcVocabFile);
      srcVocab{end+1} = '<eos>';
      params.srcVocabSize = length(srcVocab);
    end
    
    [tgtVocab] = loadVocab(params.tgtVocabFile);    
    tgtVocab{end+1} = '<eos>';
    params.tgtVocabSize = length(tgtVocab);
  end
  if params.isBi
    params.srcEos = params.srcVocabSize;
  end
  params.tgtEos = params.tgtVocabSize;
 
  %%% INIT MODEL PARAMETERS %%%
  [model, params] = init(params);
  
  % print params
  printParams(params);

  %%% CHECK GRAD %%%
  if params.isGradCheck
    % generate pseudo data
    srcTrainSents = cell(1, params.batchSize);
    tgtTrainSents = cell(1, params.batchSize);
    for ii=1:params.batchSize
      srcTrainMaxLen = 5;
      tgtTrainMaxLen = 5;

      srcLen = randi([1, srcTrainMaxLen-1]);
      srcTrainSents{ii} = randi([1, params.srcVocabSize-1], 1, srcLen);
      srcTrainSents{ii}(end+1) = params.srcEos;

      tgtLen = randi([1, tgtTrainMaxLen-1]);
      tgtTrainSents{ii} = randi([1, params.tgtVocabSize-1], 1, tgtLen); 
      tgtTrainSents{ii}(end+1) = params.tgtEos;
    end
    
    % prepare data
    [inputTrain, inputTrainMask, tgtTrainOutput, tgtTrainMask, srcTrainMaxLen, tgtTrainMaxLen] = prepare_data(srcTrainSents, tgtTrainSents, params);
    
    % check grad
    tic
    gradCheck(model, inputTrain, inputTrainMask, tgtTrainOutput, tgtTrainMask, srcTrainMaxLen, tgtTrainMaxLen, params);
    toc 
    return;
  end
  
  
  %% load valid & test data
  [inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, numValidWords] = loadPrepareData(params, params.validPrefix, srcVocab, tgtVocab);
  [inputTest, inputTestMask, tgtTestOutput, tgtTestMask, srcTestMaxLen, tgtTestMaxLen, numTestWords] = loadPrepareData(params, params.testPrefix, srcVocab, tgtVocab);
  
  %% load train data
  tgtTrainFile = sprintf('%s.%s', params.trainPrefix, params.tgtLang);
  if params.isBi
    srcTrainFile = sprintf('%s.%s', params.trainPrefix, params.srcLang);
    fprintf(2, '# Load train using srcFile "%s" and tgtFile "%s"\n', srcTrainFile, tgtTrainFile);
  end
  chunkSize = params.batchSize*100;
  srcID = fopen(srcTrainFile, 'r');
  tgtID = fopen(tgtTrainFile, 'r');
  [srcTrainSents, numTrainSents] = loadBatchData(srcID, params.baseIndex, chunkSize, params.srcEos);
  [tgtTrainSents] = loadBatchData(tgtID, params.baseIndex, chunkSize, params.tgtEos);
  printSent(srcTrainSents{1}, srcVocab, '  src:');
  printSent(tgtTrainSents{1}, tgtVocab, '  tgt:');

  %%% TRAINING %%%
  params.lr = params.learningRate;
  params.iter = 0;  % number of batches we have processed
  params.epoch = 1;
  params.bestCostValid = 1e5;
  
  totalCost = 0; totalWords = 0;
  evalFreq = params.logFreq*10;
  modelFile = [outDir '/model'];

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
    if params.epoch==1
      params.epochBatchCount = params.epochBatchCount + numBatches;
    end
    for batchId = 1 : numBatches
      params.iter = params.iter + 1;
      startId = (batchId-1)*params.batchSize+1;
      endId = batchId*params.batchSize;
      if endId > numTrainSents
        endId = numTrainSents;
      end
      
      srcBatchSents = srcTrainSents(startId:endId);
      tgtBatchSents = tgtTrainSents(startId:endId);
      
      % prepare data
      [inputTrain, inputTrainMask, tgtTrainOutput, tgtTrainMask, srcTrainMaxLen, tgtTrainMaxLen] = prepare_data(srcBatchSents, tgtBatchSents, params);
      if params.isGPU && (srcTrainMaxLen>params.srcLenLimit || tgtTrainMaxLen>params.tgtLenLimit) % sentences are too long, skip if using GPUs
        continue;
      end
      
      % core part
      [cost, grad] = lstmCostGrad(model, inputTrain, inputTrainMask, tgtTrainOutput, tgtTrainMask, srcTrainMaxLen, tgtTrainMaxLen, params, 0);
      if isnan(cost) || isinf(cost) %~isempty(find(isnan(log_probs), 1))
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
      model.W_src = model.W_src - params.lr*scale*grad.W_src;
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

    % read more data
    [srcTrainSents, numTrainSents] = loadBatchData(srcID, params.baseIndex, chunkSize, params.srcEos);
    [tgtTrainSents] = loadBatchData(tgtID, params.baseIndex, chunkSize, params.tgtEos);
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
        [srcTrainSents, numTrainSents] = loadBatchData(srcID, params.baseIndex, chunkSize, params.srcEos);
        [tgtTrainSents] = loadBatchData(tgtID, params.baseIndex, chunkSize, params.tgtEos);
      else % done training
        fprintf(2, '# Done training, %s\n', datestr(now));
        break; 
      end
    end
  end % end for while(1)
end

function [inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen, numValidWords] = loadPrepareData(params, ...
    prefix, srcVocab, tgtVocab)
  %% load valid & test data
  tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
  if params.isBi
    srcFile = sprintf('%s.%s', prefix, params.srcLang);
    fprintf(2, '# Load valid data srcFile %s and tgtFile %s ... ', srcFile, tgtFile);
    
    % src
    [srcSents] = loadMonoData(srcFile, params.srcEos, -1, params.baseIndex);
    printSent(srcSents{1}, srcVocab, '  src:');
    
    % tgt
    [tgtSents] = loadMonoData(tgtFile, params.tgtEos, -1, params.baseIndex);
    printSent(tgtSents{1}, tgtVocab, '  tgt:');
    
    [inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen] = prepare_data(srcSents, tgtSents, params);
  else
    [tgtSents] = loadMonoData(tgtFile, tgtEos, -1, params.baseIndex);
    printSent(tgtSents{1}, tgtVocab, '  tgt:');
  
    fprintf(2, '# Load valid data tgtFile %s ... ', tgtFile);
    [inputValid, inputValidMask, tgtValidOutput, tgtValidMask, srcValidMaxLen, tgtValidMaxLen] = prepare_mono_data(tgtSents, params);
  end
  numValidWords = sum(sum(tgtValidMask));
  fprintf(2, 'numWords=%d\n', numValidWords);
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

%% Init model parameters %%
function [model, params] = init(params)
  % special zero words at the end of each vocab
  % in which the emb is all zero and we will never back prob
  % stack vocab:  tgt-vocab + src-vocab
  params.tgtZeroId = params.tgtEos; 
  if params.isBi
    params.srcVocabSize = params.srcVocabSize + 1;
    params.srcZeroId = params.srcVocabSize + params.tgtVocabSize;
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
    model.W_emb(:, params.srcZeroId) = zeros(params.lstmSize, 1);
  end
  model.W_emb(:, params.tgtZeroId) = zeros(params.lstmSize, 1);
end

%% Prepare data %%
%  organize data into matrix format and produce masks, add tgtVocabSize to srcSents:
%   input:      numSents * (srcMaxLen+tgtMaxLen-1)
%   tgtOutput: numSents * tgtMaxLen
%   tgtMask  : numSents * tgtMaxLen, indicate where to ignore in the tgtOutput
function [input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen] = prepare_data(srcSents, tgtSents, params)
  src_eos = srcSents{1}(end) + params.tgtVocabSize;
  numSents = length(srcSents);
  srcMaxLen = max(cellfun(@(x) length(x), srcSents));
  tgtMaxLen = max(cellfun(@(x) length(x), tgtSents));
  input = [params.srcZeroId*ones(numSents, srcMaxLen) params.tgtZeroId*ones(numSents, tgtMaxLen-1)];
  tgtOutput = params.tgtZeroId*ones(numSents, tgtMaxLen);
  for ii=1:numSents
    src_len = length(srcSents{ii});
    tgt_len = length(tgtSents{ii});
    input(ii, srcMaxLen-src_len+1:srcMaxLen) = srcSents{ii} + params.tgtVocabSize; % src part
    input(ii, srcMaxLen+1:srcMaxLen+tgt_len-1) = tgtSents{ii}(1:end-1); % tgt part
    tgtOutput(ii, 1:tgt_len) = tgtSents{ii};
  end
  tgtMask = (tgtOutput~=params.tgtZeroId);
  inputMask = (input~=params.srcZeroId & input~=params.tgtZeroId);
  
  % the last src symbol needs to be eos for all sentences
  assert(length(unique(input(:, srcMaxLen)))==1); 
  assert(input(1, srcMaxLen)==src_eos);
end

%% Check gradients %%
function gradCheck(model, input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params)
  [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb);
  num_params = length(theta);
  fprintf(2, '# Num params=%d\n', num_params);
  [totalCost, grad] = lstmCostGrad(model, input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params, 0);
  
  theoretical_grad =  param2stack(grad.W_src, grad.W_tgt, grad.W_soft, full(grad.W_emb));
  empirical_grad = zeros(num_params, 1);
  delta = 0.0001;
  abs_diff = 0;
  local_abs_diff = 0;
  for i=1:num_params
    theta_new = theta;
    theta_new(i) = theta_new(i) + delta;
    [model_new.W_src, model_new.W_tgt, model_new.W_soft, model_new.W_emb] = stack2param(theta_new, decodeInfo);
    totalCost_new = lstmCostGrad(model_new, input, inputMask, tgtOutput, tgtMask, srcMaxLen, tgtMaxLen, params, 0);
    empirical_grad(i) = (totalCost_new-totalCost)/delta;
    abs_diff = abs_diff + abs(empirical_grad(i)-theoretical_grad(i));
    local_abs_diff = local_abs_diff + abs(empirical_grad(i)-theoretical_grad(i));
    if i==1
      fprintf(2, '# W_src\n');
    end
    if i==numel(model.W_src) + 1
      fprintf(2, '  local_diff=%g\n', local_abs_diff);
      local_abs_diff = 0;
      fprintf(2, '# W_tgt\n');
    end
    if i==numel(model.W_src) + numel(model.W_tgt) + 1
      fprintf(2, '  local_diff=%g\n', local_abs_diff);
      local_abs_diff = 0;
      fprintf(2, '# W_soft\n');
    end
    if i==numel(model.W_src) + numel(model.W_tgt) + numel(model.W_soft) + 1
      fprintf(2, '  local_diff=%g\n', local_abs_diff);
      local_abs_diff = 0;
      fprintf(2, '# W_emb\n');
    end
    fprintf(2, '%10.6f\t%10.6f\tdiff=%g\n', empirical_grad(i), theoretical_grad(i), abs(empirical_grad(i)-theoretical_grad(i))); % \tcost_new=%g\tcost=%g, totalCost_new, totalCost
  end
  fprintf(2, '  local_diff=%g\n', local_abs_diff);
  fprintf(2, '# Num params=%d, abs_diff=%g\n', num_params, abs_diff);
end

%% Load parallel sentences %%
% function [srcSents, tgtSents, srcNumSents] = loadParallelData(srcFile, tgtFile, srcEos, tgtEos, numSents, baseIndex)
%   [srcSents, srcNumSents] = loadMonoData(srcFile, srcEos, numSents, baseIndex);
%   [tgtSents, tgtNumSents] = loadMonoData(tgtFile, tgtEos, numSents, baseIndex);
%   assert(srcNumSents==tgtNumSents);
% end
