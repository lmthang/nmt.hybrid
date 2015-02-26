function [cost, softmaxGrad, grad_ht] = softmaxCostGrad(matrixName, h_t, t, tgtPredictedWords, model, params, trainData, curMask, numClasses)
%%%
%
% Perform softmax prediction and backprop.
%   matrixName: 'W_soft' for predicting words (mostly used), and
%               'W_softPos' for predicting positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  mask = curMask.mask;
  unmaskedIds = curMask.unmaskedIds;
  
  % lstm hiddent state to softmax
  if params.attnFunc>0 % attention mechanism
    [softmax_h, attn_h_concat, alignWeights, alignScores, attnInput] = lstm2softHid(h_t, params, model, trainData, curMask);
  else
    [softmax_h] = lstm2softHid(h_t, params, model);
  end

  if numClasses == 0 % normal softmax
    [probs, scores, norms] = softmax(model.(matrixName)*softmax_h, mask);
    
      % assert
    if params.assert
      if params.isGPU
        assert(gather(sum(sum(abs(scores(:, curMask.maskedIds)))))==0);
      else
        assert(sum(sum(abs(scores(:, curMask.maskedIds))))==0);
      end
    end

    % cost
    tgtPredictedWords = tgtPredictedWords(unmaskedIds);
    scoreIndices = sub2ind(size(scores), tgtPredictedWords, unmaskedIds); % 1 * length(tgtPredictedWords)
    cost = - sum(scores(scoreIndices)) + sum(log(norms).*mask);
  else % class-based softmax
    output_class = trainData.output_class;
    output_in_class = trainData.output_in_class;
    srcMaxLen = trainData.srcMaxLen;
    curBatchSize = trainData.curBatchSize;
    % class loss
    [class_probs, class_scores, class_norms] = softmax(model.W_soft_class*softmax_h, mask);
    tgt_predicted_classes = output_class(curMask.unmaskedIds, t-srcMaxLen+1)'; % predict output class
    class_score_indices = sub2ind([params.numClasses, curBatchSize], tgt_predicted_classes, curMask.unmaskedIds);
    class_cost = - sum(class_scores(class_score_indices)) + sum(log(class_norms).*curMask.mask);
    
    % in class loss
    curr_class = output_class(:,t-srcMaxLen+1);
    in_class_raws = sum(bsxfun(@times, model.W_soft_inclass(curr_class,:,:), permute(softmax_h, [2 3 1])), 3)';
    [in_class_probs, in_class_scores, in_class_norms] = softmax(in_class_raws, curMask.mask);
    tgt_predicted_in_class = output_in_class(curMask.unmaskedIds, t-srcMaxLen+1)'; % predict output in class
    in_class_score_indices = sub2ind([params.class_size, curBatchSize], tgt_predicted_in_class, curMask.unmaskedIds);
    in_class_cost = - sum(in_class_scores(in_class_score_indices)) + sum(log(in_class_norms).*curMask.mask);
    
    cost = class_cost + in_class_cost;
  end
  
  % grad
  if trainData.isTest==0 % compute grad
    if numClasses == 0 % normal softmax
      probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words

      % softmax_h
      grad_softmax_h = model.(matrixName)'* probs;
    else
      class_probs(class_score_indices) = class_probs(class_score_indices) - 1;
      in_class_probs(in_class_score_indices) = in_class_probs(in_class_score_indices) - 1;

      % grad_softmax_h
      grad_softmax_h = model.W_soft_class'*class_probs;
      grad_softmax_h = grad_softmax_h + sum(bsxfun(@times, permute(model.W_soft_inclass(curr_class,:,:),[1 3 2]), permute(in_class_probs, [2 3 1])), 3)';
    end

    % softmax -> h_t
    if params.softmaxDim>0 || params.attnFunc>0 % softmax compression or attention
      if params.softmaxDim>0 % f(W_h * h_t)
        % f'(softmax_h).*grad_softmax_h
        tmpResult = params.nonlinear_f_prime(softmax_h).*grad_softmax_h;

        % grad.W_h
        softmaxGrad.W_h = tmpResult*h_t';

        % grad_ht
        grad_ht = model.W_h'*tmpResult;
      elseif params.attnFunc>0 % f(W_ah*[attn_t; tgt_h_t])
        [softmaxGrad, grad_ht] = attnBackprop(model, trainData.srcHidVecs, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, alignScores, attnInput, params);
      end
    else % normal softmax
      grad_ht = grad_softmax_h;
    end
    
    % W_soft
    if numClasses>0 % class-based softmax
      % grad.W_soft_class
      softmaxGrad.W_soft_class = class_probs*softmax_h';
      softmaxGrad.W_soft_inclass = zeroMatrix([params.numClasses params.class_size params.lstmSize], params.isGPU, params.dataType);
      add = bsxfun(@times, permute(in_class_probs,[2 1 3]), permute(softmax_h,[2 3 1]));
      add = reshape(add, [size(add,1), size(add,2)*size(add,3)])';
      [accum, idx] = aggregateMatrix(add, curr_class, params.isGPU, params.dataType);
      softmaxGrad.W_soft_inclass(idx,:,:) = reshape(accum', [length(idx) params.class_size params.lstmSize]);
    else % normal softmax
      softmaxGrad.(matrixName) = probs*softmax_h';
    end
  else
    softmaxGrad = [];
    grad_ht = [];
  end % end isTest
end
