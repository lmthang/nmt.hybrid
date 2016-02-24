function [dc, dh, grad_W_rnn, grad_W_emb, grad_emb_indices, attnGrad, grad_srcHidVecs_total, srcCharGrad] = rnnLayerBackprop(W_rnn, rnnStates, ...
  initState, top_grads, dc, dh, input, masks, params, rnnFlags, attnInfos, trainData, model, tgtCharGrad)
% Running Multi-layer RNN for one time step.
% Input:
%   W_rnn: recurrent connections of multiple layers, e.g., W_rnn{ll}.
%   initState: begin state
%   input: indices for the current batch
%   isDecoder: 1 -- on the decoder side
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

T = size(input, 2);

% emb
totalWordCount = params.curBatchSize * T;
allEmbGrads = zeroMatrix([params.lstmSize, totalWordCount], params.isGPU, params.dataType);
allEmbIndices = zeros(totalWordCount, 1);
wordCount = 0;

% attention
if rnnFlags.attn && rnnFlags.decode
  grad_srcHidVecs_total = zeroMatrix([params.lstmSize, params.curBatchSize, params.numSrcHidVecs], params.isGPU, params.dataType);
else
  grad_srcHidVecs_total = [];
  attnGrad = [];
end

srcCharGrad = [];

% masks
[maskInfos] = prepareMask(masks);

if ~isempty(tgtCharGrad)
  rareCountRemain = tgtCharGrad.numRareWords;
end
rareAttnInputGrad_char = [];
rareIndices = [];
for tt=T:-1:1 % time
  % attention
  if rnnFlags.attn && rnnFlags.decode
    % char
    % TODO
    
    if rnnFlags.charTgtGen && tgtCharGrad.numRareWords>0 && ~isempty(tgtCharGrad.initAttnInput)
      rareIndices = find(tgtCharGrad.rareFlags(:, tt));
      rareAttnInputGrad_char = tgtCharGrad.initAttnInput(:, rareCountRemain-length(rareIndices)+1:rareCountRemain);
      rareCountRemain = rareCountRemain - length(rareIndices);
    end
    
    % attention: softmax_h -> h_t
    h2sInfo = attnInfos{tt};
    [cur_top_grad, attnStepGrad, grad_srcHidVecs] = attnLayerBackprop(model, top_grads{tt}, trainData, h2sInfo, params, maskInfos{tt}, ...
      rareAttnInputGrad_char, rareIndices);
    fields = fieldnames(attnStepGrad);
    for ii=1:length(fields)
      field = fields{ii};
      if tt==T
        attnGrad.(field) = attnStepGrad.(field);
      else
        attnGrad.(field) = attnGrad.(field) + attnStepGrad.(field);
      end
    end

    % srcHidVecs
    if params.attnGlobal 
      grad_srcHidVecs_total = grad_srcHidVecs_total + grad_srcHidVecs;
    else
      grad_srcHidVecs_total = reshape(grad_srcHidVecs_total, params.lstmSize, []);
      grad_srcHidVecs_total(:, h2sInfo.linearIdAll) = grad_srcHidVecs_total(:, h2sInfo.linearIdAll) + grad_srcHidVecs(:, h2sInfo.linearIdSub);
      grad_srcHidVecs_total = reshape(grad_srcHidVecs_total, [params.lstmSize, params.curBatchSize, params.numSrcHidVecs]);
    end
  else % non-attention
    cur_top_grad = top_grads{tt};
  end

  if tt>1
    prevState = rnnStates{tt-1};
  else
    prevState = initState;
  end

  %% multi-layer RNN backprop
  [dc, dh, d_emb, d_W_rnn, d_feed_input] = rnnStepLayerBackprop(W_rnn, prevState, rnnStates{tt}, cur_top_grad, dc, dh, maskInfos{tt}, ...
    params, rnnFlags.feedInput);

  % recurrent grad
  for ll=params.numLayers:-1:1 % layer
    if tt==T
      grad_W_rnn{ll} = d_W_rnn{ll};
    else
      grad_W_rnn{ll} = grad_W_rnn{ll} + d_W_rnn{ll};
    end
  end

  % softmax feedinput, bottom grad send back to top grad in the previous
  % time step
  if rnnFlags.feedInput && tt>1
    top_grads{tt-1} = top_grads{tt-1} + d_feed_input;
  end

  assert(size(d_emb, 1) == params.lstmSize);
  
  % emb grad
  unmaskedIds = maskInfos{tt}.unmaskedIds;
  numWords = length(unmaskedIds);
  allEmbIndices(wordCount+1:wordCount+numWords) = input(unmaskedIds, tt);
  allEmbGrads(:, wordCount+1:wordCount+numWords) = d_emb(:, unmaskedIds);
  wordCount = wordCount + numWords;
end % end for time

allEmbGrads(:, wordCount+1:end) = [];
allEmbIndices(wordCount+1:end) = [];
[grad_W_emb, grad_emb_indices] = aggregateMatrix(allEmbGrads, allEmbIndices, params.isGPU, params.dataType);

if rnnFlags.charSrcRep && rnnFlags.decode == 0 % char src representations
  rareFlags = grad_emb_indices > params.srcCharShortList;
  srcCharGrad.embs = grad_W_emb(:, rareFlags);
  srcCharGrad.indices = grad_emb_indices(rareFlags);
  
  % frequent
  grad_W_emb = grad_W_emb(:, ~rareFlags);
  grad_emb_indices = grad_emb_indices(~rareFlags);
end
