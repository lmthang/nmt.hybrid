%%%
%
% Attentional Layer Backprop from softmax hidden state to lstm hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_h, trainData, attnInfo, params, curMask, ...
  rareAttnInputGrad_char, rareIndices)
  % softmax_h -> h_t
  [grad_input, attnGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, attnInfo.input, params.nonlinear_f_prime, attnInfo.softmax_h);

  % char
  if ~isempty(rareAttnInputGrad_char)
    grad_input(:, rareIndices) = grad_input(:, rareIndices) + rareAttnInputGrad_char;
  end
  
  if params.attnGlobal % global attention
    srcHidVecs = trainData.srcHidVecsOrig;
  else % local attention
    % TODO: if our GPUs have lots of memory, then we don't have to
    % regenerate srcHidVecs again :) Unfortuntely not!
    if ~isempty(attnInfo.linearIdSub)
      srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize*params.numAttnPositions], params.isGPU, params.dataType);
      srcHidVecsOrigReshape = reshape(trainData.srcHidVecsOrig, params.lstmSize, []);
      srcHidVecs(:, attnInfo.linearIdSub) = srcHidVecsOrigReshape(:, attnInfo.linearIdAll);
      srcHidVecs = reshape(srcHidVecs, [params.lstmSize, params.curBatchSize, params.numAttnPositions]);
    else
      srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
    end
  end

  % grad_contextVecs -> grad_srcHidVecs, grad_alignWeights
  grad_contextVecs = grad_input(1:params.lstmSize, :);
  [grad_alignWeights, grad_srcHidVecs] = contextLayerBackprop(grad_contextVecs, attnInfo.alignWeights, srcHidVecs, curMask.unmaskedIds, params);

  % local attention, grad_alignWeights -> grad_distWeights, grad_preAlignWeights
  if params.attnGlobal == 0 && params.normLocalAttn % new approach for local attention after EMNLP'15
    [grad_unNormAlignWeights] = normLayerBackprop(grad_alignWeights, attnInfo.alignWeights); %, h2sInfo.srcMaskedIds, params);
    grad_distWeights = grad_unNormAlignWeights.*attnInfo.alignScores;
    grad_scores = grad_unNormAlignWeights.*attnInfo.distWeights; % grad_preAlignWeights
  else
    if params.attnGlobal == 0
      % IMPORTANT: don't change the order of these lines
      grad_distWeights = grad_alignWeights.*attnInfo.preAlignWeights;
      grad_alignWeights = grad_alignWeights.*attnInfo.distWeights; % grad_preAlignWeights

      attnInfo.alignWeights = attnInfo.preAlignWeights;
    end

    % grad_alignWeights -> grad_scores
    [grad_scores] = normLayerBackprop(grad_alignWeights, attnInfo.alignWeights); %, attnInfo.srcMaskedIds, params);
  end
  
  if params.assert
    assert(computeSum(grad_scores(attnInfo.srcMaskedIds), params.isGPU)==0);
  end
  
  % grad_scores -> grad_ht, grad_W_a / grad_srcHidVecs
  if params.attnOpt==1
    % s_t = H_src * h_t
    [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_scores, attnInfo.h_t, srcHidVecs);
  elseif params.attnOpt==2
    % s_t = H_src * W_a * h_t
    [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_scores, attnInfo.transform_ht, srcHidVecs);
    
    % grad_h_t -> W_a * h_t
    [grad_ht, attnGrad.W_a] = linearLayerBackprop(model.W_a, grad_ht, attnInfo.h_t);
  elseif params.attnOpt==3 % s_t = softmax(v_a*f(W_a*[H_src; h_t]))
    % h2sInfo.scores = linearLayerForward(model.v_a, h2sInfo.src_ht_hid); % 1 * (curBatchSize * numAttnPositions)
    % first transpose grad_scores to curBatchSize*numPositions, then flatten
    [grad_src_ht_hid, attnGrad.v_a] = linearLayerBackprop(model.v_a, reshape(grad_scores', 1, []), attnInfo.src_ht_hid);

    grad_srcHidVecs1 = params.nonlinear_f_prime(attnInfo.src_ht_hid).*grad_src_ht_hid;
    grad_srcHidVecs1 = reshape(grad_srcHidVecs1, size(srcHidVecs)); 

    [grad_ht, attnGrad.W_a] = hiddenLayerBackprop(model.W_a, grad_src_ht_hid, reshape(repmat(attnInfo.h_t, [1, 1, params.numAttnPositions]), params.lstmSize, []), params.nonlinear_f_prime, attnInfo.src_ht_hid);
    grad_ht = sum(reshape(grad_ht, size(srcHidVecs)), 3);
  end
  grad_srcHidVecs = grad_srcHidVecs + grad_srcHidVecs1; % add to the existing grad_srcHidVecs
  
  % assert
  if params.assert
    assert(computeSum(grad_ht(:, curMask.maskedIds), params.isGPU)==0);
    assert(computeSum(grad_input(params.lstmSize+1:end, curMask.maskedIds), params.isGPU)==0);
  end

  % grad_ht
  grad_ht = grad_ht + grad_input(params.lstmSize+1:end, :);
  
  % local attention
  if params.attnGlobal == 0
    % grad_distWeights -> grad_mu
    [grad_mu] = distLayerBackprop(grad_distWeights, attnInfo, params);

    % grad_mu -> grad_scales
    grad_scales = (trainData.srcLens-1).*grad_mu;

    % grad_scales -> grad_ht, grad_W_pos, grad_v_pos
    [grad_ht1, attnGrad.W_pos, attnGrad.v_pos] = scaleLayerBackprop(model.W_pos, model.v_pos, grad_scales, attnInfo.h_t, attnInfo.scales, attnInfo.posForwData, params);
    grad_ht = grad_ht + grad_ht1;
  end
  
  if params.assert
    assert(computeSum(grad_ht(:, curMask.maskedIds), params.isGPU)==0);
  end
end