%%%
%
% Attentional Layer Backprop from softmax hidden state to lstm hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [grad_ht, attnGrad, grad_srcHidVecs] = attnLayerBackprop(model, grad_softmax_h, trainData, h2sInfo, params, curMask) %isPredictPos, params)
  % softmax_h -> h_t
  [grad_input, attnGrad.W_h] = hiddenLayerBackprop(model.W_h, grad_softmax_h, h2sInfo.input, params.nonlinear_f_prime, h2sInfo.softmax_h);

  if params.attnGlobal % soft attention
    srcHidVecs = trainData.absSrcHidVecs;
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
  [grad_alignWeights, grad_srcHidVecs] = contextLayerBackprop(grad_contextVecs, h2sInfo.alignWeights, srcHidVecs, h2sInfo.posMask.unmaskedIds, params);

  % grad_contextVecs -> grad_ht, grad_W_a, grad_srcHidVecs
  if params.predictPos==3
    % IMPORTANT: don't change the order of these lines
    % grad_alignWeights -> grad_distWeights, grad_preAlignWeights
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
  if params.attnOpt==0 % no src compare, s_t = W_a * h_t
    [grad_ht, attnGrad.W_a] = linearLayerBackprop(model.W_a, grad_scores, h2sInfo.h_t);  
  elseif params.attnOpt>0
    if params.attnOpt==1 || params.attnOpt==2
      % attnOpt 1: s_t = H_src * h_t
      % attnOpt 2: s_t = H_src * W_a * h_t
      [grad_ht, grad_srcHidVecs1] = srcCompareLayerBackprop(grad_scores, h2sInfo, srcHidVecs);
      
      % grad_h_t -> W_a * h_t
      if params.attnOpt==2
        [grad_ht, attnGrad.W_a] = linearLayerBackprop(model.W_a, grad_ht, h2sInfo.h_t);
      end
    elseif params.attnOpt==3 % s_t = softmax(v_a*f(W_a*[H_src; h_t]))
      % h2sInfo.scores = linearLayerForward(model.v_a, h2sInfo.src_ht_hid); % 1 * (curBatchSize * numAttnPositions)
      % first transpose grad_scores to curBatchSize*numPositions, then flatten
      [grad_src_ht_hid, attnGrad.v_a] = linearLayerBackprop(model.v_a, reshape(grad_scores', 1, []), h2sInfo.src_ht_hid);
        
      % ht_transform = reshape(repmat(model.W_a*h_t, [1, 1, params.numAttnPositions]), params.lstmSize, []); % lstmSize * (curBatchSize * numSrcHidVecs)
      % h2sInfo.src_ht_hid = params.nonlinear_f(reshape(srcHidVecs, params.lstmSize, []) + ht_transform); % lstmSize * (curBatchSize * numSrcHidVecs)
      grad_srcHidVecs1 = reshape(params.nonlinear_f_prime(h2sInfo.src_ht_hid).*grad_src_ht_hid, size(srcHidVecs)); 
      
      [grad_ht, attnGrad.W_a] = hiddenLayerBackprop(model.W_a, grad_src_ht_hid, reshape(repmat(h2sInfo.h_t, [1, 1, params.numAttnPositions]), params.lstmSize, []), params.nonlinear_f_prime, h2sInfo.src_ht_hid);
      grad_ht = sum(reshape(grad_ht, size(srcHidVecs)), 3);
      
%       % h2sInfo.srcHidVecs_transform = model.W_a_src*reshape(srcHidVecs, params.lstmSize, []);
%       % h2sInfo.ht_transform = model.W_a_tgt*h_t;
%       % h2sInfo.src_ht_hid = params.nonlinear_f(h2sInfo.srcHidVecs_transform + repmat(h2sInfo.ht_transform, [1, 1, params.numAttnPositions]));
%       [grad_srcHidVecs1, attnGrad.W_a_src] = hiddenLayerBackprop(model.W_a_src, grad_src_ht_hid, reshape(srcHidVecs, params.lstmSize, []), params.nonlinear_f_prime, h2sInfo.src_ht_hid);
%        grad_srcHidVecs1 = reshape(grad_srcHidVecs1, size(srcHidVecs));
%       
%       [grad_ht, attnGrad.W_a_tgt] = hiddenLayerBackprop(model.W_a_tgt, grad_src_ht_hid, reshape(repmat(h2sInfo.h_t, [1, 1, params.numAttnPositions]), params.lstmSize, []), params.nonlinear_f_prime, h2sInfo.src_ht_hid);
%       grad_ht = sum(reshape(grad_ht, size(srcHidVecs)), 3);
      
%       % assert
%       if params.assert
%         % h2sInfo.src_ht_hid = hiddenLayerForward(model.W_a, h2sInfo.src_ht_concat, params.nonlinear_f); % lstmSize * (curBatchSize * numAttnPositions)
%         [grad_src_ht_concat, attnGrad.W_a] = hiddenLayerBackprop([model.W_a_src model.W_a_tgt], grad_src_ht_hid, h2sInfo.src_ht_concat, params.nonlinear_f_prime, h2sInfo.src_ht_hid);
%       
%         % h2sInfo.src_ht_concat = [reshape(srcHidVecs, params.lstmSize, []); reshape(repmat(h_t, [1, 1, params.numAttnPositions]), params.lstmSize, [])];
%         grad_srcHidVecs2 = reshape(grad_src_ht_concat(1:params.lstmSize, :), size(srcHidVecs));
%         grad_ht2 = sum(reshape(grad_src_ht_concat(params.lstmSize+1:end, :), size(srcHidVecs)), 3);
%         
%         assert(computeSum(grad_srcHidVecs2-grad_srcHidVecs1, params.isGPU)<1e-10);
%         assert(computeSum(grad_ht2-grad_ht, params.isGPU)<1e-10);
%       end
    end
    
    grad_srcHidVecs = grad_srcHidVecs + grad_srcHidVecs1; % add to the existing grad_srcHidVecs
  end
  
  % assert
  if params.assert
    assert(computeSum(grad_ht(:, curMask.maskedIds), params.isGPU)==0);
    assert(computeSum(grad_input(params.lstmSize+1:end, curMask.maskedIds), params.isGPU)==0);
  end

  % grad_ht
  grad_ht = grad_ht + grad_input(params.lstmSize+1:end, :);
  
  
  if params.predictPos==3
    % since linearIdSub is for matrix of size [curBatchSize, numAttnPositions], 
    % we need to transpose grad_alignWeights to be of that size.
    grad_distWeights = grad_distWeights';
    h2sInfo.distWeights = h2sInfo.distWeights';

    % grad_distWeights -> grad_mu
    [grad_mu] = distLayerBackprop(grad_distWeights, h2sInfo.distWeights, h2sInfo, params);

    % grad_mu -> grad_scales
    grad_scales = trainData.srcLens.*grad_mu;

    % grad_scales -> grad_ht, grad_W_pos, grad_v_pos
    [grad_ht1, attnGrad.W_pos, attnGrad.v_pos] = scaleLayerBackprop(model.W_pos, model.v_pos, grad_scales, h2sInfo.h_t, h2sInfo.scales, h2sInfo.posForwData, params);
    grad_ht = grad_ht + grad_ht1;
  end
end