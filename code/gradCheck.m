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

  for ii=1:params.batchSize
    if params.isBi
      srcLen = randi([1, srcTrainMaxLen-1]);
      srcTrainSents{ii} = randi([1, params.srcVocabSize-2], 1, srcLen); % exclude <s_eos> and  <s_sos>
      srcTrainSents{ii}(end+1) = params.srcEos;
    end

    tgtLen = randi([1, tgtTrainMaxLen-1]);
    if params.posModel>0 % positional models: generate pairs of pos/word
      tgtTrainSents{ii} = zeros(1, 2*tgtLen);
      tgtTrainSents{ii}(1:2:2*tgtLen-1) = randi([params.startPosId, params.tgtVocabSize-2], 1, tgtLen); % positions (exclude <p_eos> and <t_eos> at the end)
      tgtTrainSents{ii}(2:2:2*tgtLen) = randi([1, params.startPosId-1], 1, tgtLen); % words
    else
      tgtTrainSents{ii} = randi([1, params.tgtVocabSize-1], 1, tgtLen); % non-eos words
    end
    tgtTrainSents{ii}(end+1) = params.tgtEos;
  end

  % prepare data
  [trainData] = prepareData(srcTrainSents, tgtTrainSents, 0, params);
  printSent(2, trainData.input(1, :), params.vocab, '   input 1:');
  printSent(2, trainData.tgtOutput(1, :), params.vocab, '  output 1:');
  % positional models
  if params.posModel>0
    printSent(2, trainData.srcPos(1, :), params.vocab, '  srcPos 1:');
  end
    
  % for gradient check purpose
  if params.dropout<1 % use the same dropout mask
    curBatchSize = size(trainData.input, 1);
    params.dropoutMask = randSimpleMatrix([params.lstmSize curBatchSize], params.isGPU, params.dataType)/params.dropout;
    
    if params.posModel>0
      params.dropoutMaskPos = randSimpleMatrix([2*params.lstmSize curBatchSize], params.isGPU, params.dataType)/params.dropout;
    end
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

%   % theta
%   [theta, decodeInfo] = struct2vec(model, params.vars);
%   numParams = length(theta);
%   fprintf(2, '# Num params=%d\n', numParams);
  
  
%grad = struct2vec(grad, params.vars);
%   empGrad = zeros(numParams, 1);
%   numSrcParams = 0;
%   for ii=1:length(model.W_src)
%     numSrcParams = numSrcParams + numel(model.W_src{ii});
%   end
%   numTgtParams = 0;
%   for ii=1:length(model.W_tgt)
%     numTgtParams = numTgtParams + numel(model.W_tgt{ii});
%   end
%   for i=1:numParams
%     thetaNew = theta;
%     thetaNew(i) = thetaNew(i) + delta;
%     [modelNew] = vec2struct(thetaNew, decodeInfo);
%     totalCost_new = lstmCostGrad(modelNew, trainData, params, 0);
%     empGrad(i) = (totalCost_new-totalCost)/delta;
%     abs_diff = abs_diff + abs(empGrad(i)-anaGrad(i));
%     local_abs_diff = local_abs_diff + abs(empGrad(i)-anaGrad(i));
%     if params.isBi
%       if i==1
%         fprintf(2, '# W_src\n');
%       end
%       if i==numSrcParams + 1
%         fprintf(2, '  local_diff=%g\n', local_abs_diff);
%         local_abs_diff = 0;
%         fprintf(2, '# W_tgt\n');
%       end
%     else
%       if i==1
%         fprintf(2, '# W_tgt\n');
%       end
%     end
%     
%     % W_soft
%     if i==numSrcParams + numTgtParams + 1
%       fprintf(2, '  local_diff=%g\n', local_abs_diff);
%       local_abs_diff = 0;
%       fprintf(2, '# W_soft [%d, %d]\n', size(model.W_soft, 1), size(model.W_soft, 2));
%     end
%     
%     % W_emb
%     if i==numSrcParams + numTgtParams + numel(model.W_soft) + 1
%       fprintf(2, '  local_diff=%g\n', local_abs_diff);
%       local_abs_diff = 0;
%       fprintf(2, '# W_emb [%d, %d]\n', size(model.W_emb, 1), size(model.W_emb, 2));
%     end
%     
%     % W_h
%     if params.softmaxDim>0
%       if i==numSrcParams + numTgtParams + numel(model.W_soft) + numel(model.W_emb) + 1
%         fprintf(2, '  local_diff=%g\n', local_abs_diff);
%         local_abs_diff = 0;
%         fprintf(2, '# W_h [%d, %d]\n', size(model.W_h, 1), size(model.W_h, 2));
%       end
%     end

%   if params.isBi
%     if params.attnFunc==0
%       if params.softmaxDim>0
%         [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb, model.W_h);
%       else
%         [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb);
%       end
%       
%     elseif params.attnFunc==1
%       [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb, model.W_a);
%     elseif params.attnFunc==2
%       [theta, decodeInfo] = param2stack(model.W_src, model.W_tgt, model.W_soft, model.W_emb, model.W_a, model.W_a_tgt, model.v_a);
%     end
%   else
%     [theta, decodeInfo] = param2stack(model.W_tgt, model.W_soft, model.W_emb);
%   end


%   if params.isBi
%     if params.attnFunc==0
%       if params.softmaxDim>0
%         anaGrad =  param2stack(grad.W_src, grad.W_tgt, grad.W_soft, grad.W_emb, grad.W_h);
%       else
%         anaGrad =  param2stack(grad.W_src, grad.W_tgt, grad.W_soft, grad.W_emb);
%       end
%     elseif params.attnFunc==1
%       anaGrad =  param2stack(grad.W_src, grad.W_tgt, grad.W_soft, grad.W_emb, grad.W_a);
%     elseif params.attnFunc==2
%       anaGrad =  param2stack(grad.W_src, grad.W_tgt, grad.W_soft, grad.W_emb, grad.W_a, grad.W_a_tgt, grad.v_a);
%     end
%   else
%     anaGrad =  param2stack(grad.W_tgt, grad.W_soft, grad.W_emb);
%   end

%     if params.isBi
%       [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb] = stack2param(thetaNew, decodeInfo);
%       if params.attnFunc==0
%         if params.softmaxDim>0
%           [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb, modelNew.W_h] = stack2param(thetaNew, decodeInfo);
%         else
%           [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb] = stack2param(thetaNew, decodeInfo);
%         end
%       elseif params.attnFunc==1
%         [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb, modelNew.W_a] = stack2param(thetaNew, decodeInfo);
%       elseif params.attnFunc==2
%         [modelNew.W_src, modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb, modelNew.W_a, modelNew.W_a_tgt, modelNew.v_a] = stack2param(thetaNew, decodeInfo);
%       end
%     else
%       model.W_src = [];
%       [modelNew.W_tgt, modelNew.W_soft, modelNew.W_emb] = stack2param(thetaNew, decodeInfo);
%     end
