function gradCheck(model, params)
%%%
%
% Perform gradient check.
%
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
%%%
  % generate pseudo data
  if params.isBi
    srcTrainMaxLen = params.maxSentLen;
    srcTrainSents = cell(1, params.batchSize);
  else
    srcTrainSents = {};
  end

  tgtTrainSents = cell(1, params.batchSize);
  tgtTrainMaxLen = params.maxSentLen;

  % generate src/tgt sentences (do not generate <eos> symbol)
  for ii=1:params.batchSize
    if params.isBi
      srcLen = randi([1, srcTrainMaxLen-1]);
      srcTrainSents{ii} = randi([1, params.srcVocabSize-2], 1, srcLen); % exclude <s_eos> and  <s_zero>
    end

    tgtLen = randi([1, tgtTrainMaxLen-1]);
    tgtTrainSents{ii} = randi([1, params.tgtVocabSize-2], 1, tgtLen); % exclude <t_sos> and <t_eos>

    % positional models: generate pairs of pos/word
    if params.posModel>0 
      tgtTrainSents{ii} = zeros(1, 2*tgtLen);
      tgtTrainSents{ii}(1:2:2*tgtLen-1) = randi([params.startPosId, params.startPosId + params.posVocabSize-2], 1, tgtLen); % positions (exclude <t_eos> at the end)
      tgtTrainSents{ii}(2:2:2*tgtLen) = randi([1, params.startPosId-1], 1, tgtLen); % words
    else
    end
  end

  % prepare data
  [trainData] = prepareData(srcTrainSents, tgtTrainSents, 0, params);
  printSent(2, trainData.srcInput(1, :), params.vocab, 'src input:');
  printSent(2, trainData.tgtOutput(1, :), params.vocab, 'tgt output:');
  fprintf(2, 'src mask: %s\n', num2str(trainData.srcMask(1, :)));
  fprintf(2, 'tgt mask: %s\n', num2str(trainData.tgtMask(1, :)));
  
  % positional models
  if params.posModel>0
    printSent(2, trainData.posOutput(1, :), params.vocab, '  pos output:');
  end
    
  % for gradient check purpose
  if params.dropout<1 % use the same dropout mask
    curBatchSize = size(trainData.input, 1);
    params.dropoutMask = randSimpleMatrix([params.lstmSize curBatchSize], params.isGPU, params.dataType)/params.dropout;
    
%     if params.posModel>0
%       params.dropoutMaskPos = randSimpleMatrix([2*params.lstmSize curBatchSize], params.isGPU, params.dataType)/params.dropout;
%     end
  end
  
  % analytic grad
  [costs, grad] = lstmCostGrad(model, trainData, params, 0);
  totalCost = costs.total;
  
  % W_emb
  full_grad_W_emb = zeroMatrix(size(model.W_emb), params.isGPU, params.dataType);
  if params.isGPU
    full_grad_W_emb(:, grad.indices) = gather(grad.W_emb);
  else
    full_grad_W_emb(:, grad.indices) = grad.W_emb;
  end
  grad.W_emb = full_grad_W_emb;
    
  % W_soft_inclass
  if params.numClasses>0
    full_grad_W_soft_inclass = zeroMatrix(size(model.W_soft_inclass), params.isGPU, params.dataType);
    if params.isGPU
      full_grad_W_soft_inclass(:, :, grad.classIndices) = gather(grad.W_soft_inclass);
    else
      full_grad_W_soft_inclass(:, :, grad.classIndices) = grad.W_soft_inclass;
    end
    grad.W_soft_inclass = full_grad_W_soft_inclass;
  end
  
  % empirical grad
  delta = 0.01;
  total_abs_diff = 0;
  numParams = 0;

  for ii=1:length(params.vars)
    field = params.vars{ii};
    if iscell(model.(field)) % cell
      for jj=1:length(model.(field))
        fprintf(2, '# %s{%d}, %s\n', field, jj, mat2str(size(model.(field){jj})));
        local_abs_diff = 0;
        for kk=1:numel(model.(field){jj})
          modelNew = model;
          modelNew.(field){jj}(kk) = modelNew.(field){jj}(kk) + delta;
          
          costs_new = lstmCostGrad(modelNew, trainData, params, 0);
          totalCost_new = costs_new.total;
          empGrad = (totalCost_new-totalCost)/delta;
          
          anaGrad = grad.(field){jj}(kk);
          abs_diff = abs(empGrad-anaGrad);
          local_abs_diff = local_abs_diff + abs_diff;
          numParams = numParams + 1;
          fprintf(2, '%10.6f\t%10.6f\tdiff=%g\n', empGrad, anaGrad, abs_diff);
        end
        total_abs_diff = total_abs_diff + local_abs_diff;
        fprintf(2, '  local_diff=%g\n', local_abs_diff);
      end
    else
      fprintf(2, '# %s, %s\n', field, mat2str(size(model.(field))));
      local_abs_diff = 0;
      for kk=1:numel(model.(field))
        modelNew = model;
        modelNew.(field)(kk) = modelNew.(field)(kk) + delta;

        costs_new = lstmCostGrad(modelNew, trainData, params, 0);
        totalCost_new = costs_new.total;
        empGrad = (totalCost_new-totalCost)/delta;

        anaGrad = grad.(field)(kk);
        abs_diff = abs(empGrad-anaGrad);
        local_abs_diff = local_abs_diff + abs_diff;
        fprintf(2, '%10.6f\t%10.6f\tdiff=%g\n', empGrad, anaGrad, abs_diff);
        numParams = numParams + 1;
      end
      total_abs_diff = total_abs_diff + local_abs_diff;
      fprintf(2, '  local_diff=%g\n', local_abs_diff);
    end 
  end
  
  fprintf(2, '# Num params=%d, abs_diff=%g\n', numParams, total_abs_diff);
end
