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
    srcTrainMaxLen = 5;
    srcTrainSents = cell(1, params.batchSize);
  else
    srcTrainSents = {};
  end

  tgtTrainSents = cell(1, params.batchSize);
  tgtTrainMaxLen = 5;

  for ii=1:params.batchSize
    if params.isBi
      srcLen = randi([1, srcTrainMaxLen-1]);
      srcTrainSents{ii} = randi([1, params.srcVocabSize-1], 1, srcLen);
      srcTrainSents{ii}(end+1) = params.srcEos;
    end

    tgtLen = randi([1, tgtTrainMaxLen-1]);
    tgtTrainSents{ii} = randi([1, params.tgtVocabSize-1], 1, tgtLen); 
    tgtTrainSents{ii}(end+1) = params.tgtEos;
  end

  % prepare data
  [trainData.input, trainData.inputMask, trainData.tgtOutput, trainData.srcMaxLen, trainData.tgtMaxLen, trainData.srcLens] = prepareData(srcTrainSents, tgtTrainSents, params);

  % theta
  if params.isBi
    if params.attnFunc==0
      [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb);
    elseif params.attnFunc==1
      [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb, model.W_a);
    elseif params.attnFunc==2
      [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb, model.W_a, model.W_a_tgt, model.v_a);
    end
  else
    [theta, decodeInfo] = param2stack(model.W_tgt, model.W_soft, model.W_emb);
  end
  numParams = length(theta);
  fprintf(2, '# Num params=%d\n', numParams);
  
  % analytic grad
  full_grad_W_emb = zeros(size(model.W_emb, 1), size(model.W_emb, 2));
  [totalCost, grad] = lstmCostGrad(model, trainData, params, 0);
  if params.isGPU
    full_grad_W_emb(:, grad.indices) = gather(grad.W_emb);
  else
    full_grad_W_emb(:, grad.indices) = grad.W_emb;
  end
  if params.isBi
    anaGrad =  param2stack(grad.W_src, grad.W_tgt, grad.W_soft, full_grad_W_emb); %full(grad.W_emb));
  else
    anaGrad =  param2stack(grad.W_tgt, grad.W_soft, full_grad_W_emb); %, full(grad.W_emb));
  end
  
  % empirical grad
  empGrad = zeros(numParams, 1);
  delta = 0.01;
  abs_diff = 0;
  local_abs_diff = 0;
  
  numSrcParams = 0;
  for ii=1:length(model.W_src)
    numSrcParams = numSrcParams + numel(model.W_src{ii});
  end
  numTgtParams = 0;
  for ii=1:length(model.W_tgt)
    numTgtParams = numTgtParams + numel(model.W_tgt{ii});
  end
  for i=1:numParams
    thetaNew = theta;
    thetaNew(i) = thetaNew(i) + delta;
    if params.isBi
      [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb] = stack2param(thetaNew, decodeInfo);
      if params.attnFunc==0
        [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb] = stack2param(thetaNew, decodeInfo);
      elseif params.attnFunc==1
        [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb, modelNew.W_a] = stack2param(thetaNew, decodeInfo);
      elseif params.attnFunc==2
        [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb, modelNew.W_a, modelNew.W_a_tgt, modelNew.v_a] = stack2param(thetaNew, decodeInfo);
      end
    else
      model.W_src = [];
      [modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb] = stack2param(thetaNew, decodeInfo);
    end
    totalCost_new = lstmCostGrad(modelNew, trainData, params, 0);
    empGrad(i) = (totalCost_new-totalCost)/delta;
    abs_diff = abs_diff + abs(empGrad(i)-anaGrad(i));
    local_abs_diff = local_abs_diff + abs(empGrad(i)-anaGrad(i));
    
    if params.isBi
      if i==1
        fprintf(2, '# W_src\n');
      end
      if i==numSrcParams + 1
        fprintf(2, '  local_diff=%g\n', local_abs_diff);
        local_abs_diff = 0;
        fprintf(2, '# W_tgt\n');
      end
    else
      if i==1
        fprintf(2, '# W_tgt\n');
      end
    end
    if i==numSrcParams + numTgtParams + 1
      fprintf(2, '  local_diff=%g\n', local_abs_diff);
      local_abs_diff = 0;
      fprintf(2, '# W_soft [%d, %d]\n', size(model.W_soft, 1), size(model.W_soft, 2));
    end
    if i==numSrcParams + numTgtParams + numel(model.W_soft) + 1
      fprintf(2, '  local_diff=%g\n', local_abs_diff);
      local_abs_diff = 0;
      fprintf(2, '# W_emb [%d, %d]\n', size(model.W_emb, 1), size(model.W_emb, 2));
    end
    fprintf(2, '%10.6f\t%10.6f\tdiff=%g\n', empGrad(i), anaGrad(i), abs(empGrad(i)-anaGrad(i))); % \tcost_new=%g\tcost=%g, totalCost_new, totalCost
  end
  fprintf(2, '  local_diff=%g\n', local_abs_diff);
  fprintf(2, '# Num params=%d, abs_diff=%g\n', numParams, abs_diff);
end
