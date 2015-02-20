function [cost, softmaxGrad, grad_ht] = softmaxCostGrad(W_soft, h_t, tgtPredictedWords, model, params, curData)
  mask = curData.mask;
  unmaskedIds = curData.unmaskedIds;
  
  % lstm hiddent state to softmax
  if params.attnFunc>0 % attention mechanism
    [softmax_h, attn_h_concat, alignWeights, alignScores, attnInput] = lstm2softHid(h_t, params, model, curData);
  else
    [softmax_h] = lstm2softHid(h_t, params, model);
  end
  
  [probs, scores, norms] = softmax(W_soft*softmax_h, mask);
          
  % assert
  if params.assert
    if params.isGPU
      assert(gather(sum(sum(abs(scores(:, curData.maskedIds)))))==0);
    else
      assert(sum(sum(abs(scores(:, curData.maskedIds))))==0);
    end
  end
  
  % cost
  tgtPredictedWords = tgtPredictedWords(unmaskedIds);
  scoreIndices = sub2ind(size(scores), tgtPredictedWords, unmaskedIds); % 1 * length(tgtPredictedWords)
  cost = - sum(scores(scoreIndices)) + sum(log(norms).*mask);
  
  % grad
  if curData.isTest==0 % compute grad
    probs(scoreIndices) = probs(scoreIndices) - 1; % minus one at predicted words

    % softmax_h
    grad_softmax_h = model.W_soft'* probs;

    % W_soft
    grad_W_soft = probs*softmax_h';

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
        [softmaxGrad, grad_ht] = attnBackprop(model, curData.srcAlignStates, softmax_h, grad_softmax_h, attn_h_concat, alignWeights, alignScores, attnInput, params);
      end
    else % normal softmax
      grad_ht = grad_softmax_h;
    end
    
    softmaxGrad.W_soft = grad_W_soft;
  else
    softmaxGrad = [];
    grad_ht = [];
  end % end isTest
end