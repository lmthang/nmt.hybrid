function [cost, softmaxGrad, grad_ht, classSoftmax] = softmaxCostGrad(matrixName, h_t, origPredLabels, model, params, trainData, curMask, varargin)
%%%
%
% Perform softmax prediction and backprop.
%   matrixName: 'W_soft' for predicting words (mostly used), and
%               'W_softPos' for predicting positions.
% When param.snumClasses>0: perform class-based softmax. origPredLabels are class
%   labels. Predicting class labels is like predicting words in the regular
%   softmax. Besides, we need to predict the inclass word indices.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  mask = curMask.mask;
  unmaskedIds = curMask.unmaskedIds;
  maskedIds = curMask.maskedIds;
  
  %% h_t -> softmax_h
  if params.attnFunc>0 % attention mechanism
    [softmax_h, attn_h_concat, alignWeights, alignScores, attnInput] = lstm2softHid(h_t, params, model, trainData, curMask);
  else
    [softmax_h] = lstm2softHid(h_t, params, model);
  end

  %% softmax
  [probs, scores, norms] = softmax(model.(matrixName)*softmax_h, mask);

  %% cost
  predLabels = origPredLabels(unmaskedIds);
  scoreIndices = sub2ind(size(scores), predLabels, unmaskedIds); % 1 * length(tgtPredictedWords)
  cost = - sum(scores(scoreIndices)) + sum(log(norms).*mask);
  
  %% class-based softmax
  if params.numClasses > 0
    predInClass = varargin{1};
    predInClass = predInClass(unmaskedIds);
    
    %% in class loss
    % model.W_soft_inclass(:,:,curClasses): classSize * softmaxSize * batchSize
    % softmax_h: softmaxSize * batchSize
    % inClassRaw: classSize * batchSize (sum across softmaxSize, dim 2)
    curClasses = origPredLabels;
    inClassRaw = reshape(sum(bsxfun(@times, model.W_soft_inclass(:,:,curClasses), permute(softmax_h, [3 1 2])), 2), params.classSize, trainData.curBatchSize);
    [inClassProbs, inClassScores, inClassNorms] = softmax(inClassRaw, mask);
    
    inClassScoreIndices = sub2ind(size(inClassScores), predInClass, unmaskedIds);
    inClassCost = - sum(inClassScores(inClassScoreIndices)) + sum(log(inClassNorms).*mask);
    
    cost = cost + inClassCost;
  end
  
  %% grad
  classSoftmax = [];
  softmaxGrad = [];
  grad_ht = [];
  if trainData.isTest==0 % compute grad
    %% loss -> grad_softmax_h
    probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words

    % softmax_h
    grad_softmax_h = model.(matrixName)'* probs;
    
    % class-based softmax
    if params.numClasses > 0 
      inClassProbs(inClassScoreIndices) = inClassProbs(inClassScoreIndices) - 1;

      % W_soft_inclass(:,:,curClasses): classSize * softmaxSize * batchSize
      % inClassProbs: classSize * batchSize
      % grad_softmax_h: softmaxSize * batchSize (sum across classSize, dim 1)
      grad_softmax_h = grad_softmax_h + squeeze(sum(bsxfun(@times, model.W_soft_inclass(:,:,curClasses), permute(inClassProbs, [1 3 2])), 1));
    end

    %% grad_softmax_h -> h_t
    if params.softmaxDim>0 || params.attnFunc>0 % softmax compression or attention
      if params.softmaxDim>0 % f(W_h * h_t)
        % f'(softmax_h).*grad_softmax_h
        tmpResult = params.nonlinear_f_prime(softmax_h).*grad_softmax_h;

        % grad.W_h
        softmaxGrad.W_h = tmpResult*h_t';

        % grad_ht
        grad_ht = model.W_h'*tmpResult;
      elseif params.attnFunc>0 % f(W_ah*[attn_t; tgt_h_t])
        [softmaxGrad, grad_ht] = attnBackprop(model, trainData.topHidVecs, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, alignScores, attnInput, params);
      end
    else % normal softmax
      grad_ht = grad_softmax_h;
    end
    
    %% loss -> W_soft
    softmaxGrad.(matrixName) = probs*softmax_h';
    
    if params.numClasses>0 % class-based softmax
      numWords = length(unmaskedIds);
      % inClassProbs(:, unmaskedIds): classSize * numWords
      % softmax_h(:, unmaskedIds): softmaxSize * numWords
      % W_soft_inclass: classSize * softmaxSize * numWords
      classSoftmax.W_soft_inclass = bsxfun(@times, permute(inClassProbs(:, unmaskedIds),[1 3 2]), permute(softmax_h(:, unmaskedIds),[3 1 2]));
      
      % grad.W_soft_inclass: (classSize * softmaxSize) * numWords
      classSoftmax.W_soft_inclass = reshape(classSoftmax.W_soft_inclass, [params.classSize*params.softmaxSize numWords]);
      classSoftmax.indices = curClasses(unmaskedIds);
    end
  end % end isTest
  
  
  % assert
  if params.assert
    if params.isGPU
      assert(gather(sum(sum(abs(scores(:, maskedIds)))))==0);
    else
      assert(sum(sum(abs(scores(:, maskedIds))))==0);
    end
    
    if params.numClasses > 0
      if params.isGPU
        assert(gather(sum(sum(abs(inClassScores(:, maskedIds)))))==0);
      else
        assert(sum(sum(abs(inClassScores(:, maskedIds))))==0);
      end
    end
  end
end
