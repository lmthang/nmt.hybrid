%%%
%
% Attentional Layer Backprop from softmax hidden state to lstm hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, params, curMask)
  % softmax_h -> h_t
  [grad_input, attnGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h2sInfo.input, params.nonlinear_f_prime, h2sInfo.softmax_h);

  if params.attnGlobal % soft attention
    srcHidVecs = trainData.srcHidVecsOrig;
  else % hard attention
    % TODO: if our GPUs have lots of memory, then we don't have to
    % regenerate srcHidVecs again :) Unfortuntely not!
    if ~isempty(h2sInfo.linearIdSub)
      srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize*params.numAttnPositions], params.isGPU, params.dataType);
      trainData.srcHidVecs = reshape(trainData.srcHidVecs, params.lstmSize, []);
      srcHidVecs(:, h2sInfo.linearIdSub) = trainData.srcHidVecs(:, h2sInfo.linearIdAll);
      srcHidVecs = reshape(srcHidVecs, [params.lstmSize, params.curBatchSize, params.numAttnPositions]);
    else
      srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
    end
  end

  % grad_contextVecs -> grad_srcHidVecs, grad_alignWeights
  grad_contextVecs = grad_input(1:params.lstmSize, :);
  [grad_alignWeights, grad_srcHidVecs] = contextLayerBackprop(grad_contextVecs, h2sInfo.alignWeights, srcHidVecs, curMask.unmaskedIds, params);

  % grad_alignWeights -> grad_distWeights, grad_preAlignWeights
  if params.predictPos
    % IMPORTANT: don't change the order of these lines
    grad_distWeights = grad_alignWeights.*h2sInfo.preAlignWeights;
    grad_alignWeights = grad_alignWeights.*h2sInfo.distWeights; % grad_preAlignWeights
    h2sInfo.alignWeights = h2sInfo.preAlignWeights;
  end
  
  % grad_alignWeights -> grad_scores
  [grad_scores] = normLayerBackprop(grad_alignWeights, h2sInfo.alignWeights, h2sInfo.srcMaskedIds, params);
  if params.assert
    assert(computeSum(grad_scores(h2sInfo.srcMaskedIds), params.isGPU)==0);
  end
  
  % grad_scores -> grad_ht, grad_W_a / grad_srcHidVecs
  if params.attnOpt==1
    % s_t = H_src * h_t
    [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_scores, h2sInfo.h_t, srcHidVecs);
  elseif params.attnOpt==2
    % s_t = H_src * W_a * h_t
    [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_scores, h2sInfo.transform_ht, srcHidVecs);
    
    % grad_h_t -> W_a * h_t
    [grad_ht, attnGrad.W_a] = linearLayerBackprop(model.W_a, grad_ht, h2sInfo.h_t);
  elseif params.attnOpt==3 % s_t = softmax(v_a*f(W_a*[H_src; h_t]))
    % h2sInfo.scores = linearLayerForward(model.v_a, h2sInfo.src_ht_hid); % 1 * (curBatchSize * numAttnPositions)
    % first transpose grad_scores to curBatchSize*numPositions, then flatten
    [grad_src_ht_hid, attnGrad.v_a] = linearLayerBackprop(model.v_a, reshape(grad_scores', 1, []), h2sInfo.src_ht_hid);

    grad_srcHidVecs1 = params.nonlinear_f_prime(h2sInfo.src_ht_hid).*grad_src_ht_hid;
    grad_srcHidVecs1 = reshape(grad_srcHidVecs1, size(srcHidVecs)); 

    [grad_ht, attnGrad.W_a] = hiddenLayerBackprop(model.W_a, grad_src_ht_hid, reshape(repmat(h2sInfo.h_t, [1, 1, params.numAttnPositions]), params.lstmSize, []), params.nonlinear_f_prime, h2sInfo.src_ht_hid);
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
  
  
  if params.predictPos
    % grad_distWeights -> grad_mu
    [grad_mu] = distLayerBackprop(grad_distWeights, h2sInfo, params);

    % grad_mu -> grad_scales
    grad_scales = (trainData.srcLens-1).*grad_mu;

    % grad_scales -> grad_ht, grad_W_pos, grad_v_pos
    [grad_ht1, attnGrad.W_pos, attnGrad.v_pos] = scaleLayerBackprop(model.W_pos, model.v_pos, grad_scales, h2sInfo.h_t, h2sInfo.scales, h2sInfo.posForwData, params);
    grad_ht = grad_ht + grad_ht1;
  end
end


%       % v_a * src_ht_hid
%       grad_scores = permute(grad_scores, [3, 2, 1]); % 1 * batchSize * numPositions
%       attnGrad.v_a = sum(sum(bsxfun(@times, h2sInfo.src_ht_hid, grad_scores), 2), 3); % sum along batchSize, numPositions
%       grad_src_ht_hid = bsxfun(@times, model.v_a, grad_scores); % lstmSize * batchSize * numPositions
%   
%       grad_srcHidVecs1 = params.nonlinear_f_prime(h2sInfo.src_ht_hid).*grad_src_ht_hid;
%       
%       % f'(outVec).*outGrad 
%       % lstmSize * batchSize * numPositions
%       tmpResult = params.nonlinear_f_prime(h2sInfo.src_ht_hid).*grad_src_ht_hid;  
% 
%       % grad_W
%       % h2sInfo.h_t: lstmSize * batchSize
%       % permute h_t into :1 * batchSize * 1 * lstmSize
%       attnGrad.W_a = squeeze(sum(sum(bsxfun(@times, tmpResult, permute(h2sInfo.h_t, [3, 2, 4, 1])), 2), 3));
% 
%       % inGrad
%       % model.W_a: lstmSize*lstmSize
%       % model.W_a'*tmpResult
%       % permute W_a to lstmSize * 1 * 1 * lstmSize
%       % after bxsfun multiply: we get lstmSize  * batchSize * numAttentionPositions * lstmSize
%       % sum over 1, 3 and then permute
%       grad_ht = squeeze(sum(sum(bsxfun(@times, permute(model.W_a, [1, 3, 4, 2]), tmpResult), 1), 3))';