function gradCheck(model, params)
%%%
%
% Perform gradient check.
%
% Thang Luong @ 2014, 2015 <lmthang@stanford.edu>
%
%%%
  delta = 0.01; % set to 0.01 to debug on GPU.

  % generate pseudo data
  if params.isBi
    srcTrainMaxLen = params.maxSentLen-2;
    srcTrainSents = cell(1, params.batchSize);
  else
    srcTrainSents = {};
  end

  tgtTrainSents = cell(1, params.batchSize);
  tgtTrainMaxLen = params.maxSentLen-2;

  % generate src/tgt sentences (do not generate <eos> symbol)
  for ii=1:params.batchSize
    if params.isBi
      srcLen = randi([1, srcTrainMaxLen-1]);
      srcTrainSents{ii} = randi([1, params.srcVocabSize-2], 1, srcLen) + 2; % exclude <s> and  </s>
    end

    tgtLen = randi([1, tgtTrainMaxLen-1]);
    tgtTrainSents{ii} = randi([1, params.tgtVocabSize-2], 1, tgtLen) + 2; % exclude <s> and </s>
  end

  % prepare data
  [trainData] = prepareData(srcTrainSents, tgtTrainSents, 0, params);
  printTrainBatch(trainData, params);
    
  % for gradient check purpose
  if params.dropout<1 % use the same dropout mask
    curBatchSize = size(trainData.tgtInput, 1);
    params.dropoutMask = (randMatrix([params.lstmSize curBatchSize], params.isGPU, params.dataType)<params.dropout)/params.dropout;
    
    if params.feedInput
      params.dropoutMaskInput = (randMatrix([2*params.lstmSize curBatchSize], params.isGPU, params.dataType)<params.dropout)/params.dropout;
    end
  end
  
  % analytic grad
  [costs, grad] = lstmCostGrad(model, trainData, params, 0);
  totalCost = costs.total;
  
  % W_emb
  full_grad_W_emb_src = zeroMatrix(size(model.W_emb_src), params.isGPU, params.dataType);
  full_grad_W_emb_src(:, grad.indices_src) = grad.W_emb_src;
  grad.W_emb_src = full_grad_W_emb_src;

  full_grad_W_emb_tgt = zeroMatrix(size(model.W_emb_tgt), params.isGPU, params.dataType);
  full_grad_W_emb_tgt(:, grad.indices_tgt) = grad.W_emb_tgt;
  grad.W_emb_tgt = full_grad_W_emb_tgt;
  
  % empirical grad
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
