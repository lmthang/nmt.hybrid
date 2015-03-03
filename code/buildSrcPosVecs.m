function [s_t, srcPosData] = buildSrcPosVecs(t, model, params, trainData, curMask)
%%%
%
% For positional models, generate src vectors based on the predicted positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%

  srcMaxLen = trainData.srcMaxLen;
  unmaskedIds = curMask.unmaskedIds;
  allSrcPos = trainData.srcPos;
  srcLens = trainData.srcLens;
  input = trainData.input;
  
  s_t = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  
  % get predicted positions
  tgtPos = t-srcMaxLen+1;
  predPositions = allSrcPos(unmaskedIds, tgtPos)';

  % eos
  eosIds = unmaskedIds(predPositions == params.eosPosId);
  s_t(:, eosIds) = repmat(model.W_emb(:, params.eosPosId), 1, length(eosIds));

  % pos
  posIds = unmaskedIds(predPositions ~= params.nullPosId & predPositions ~= params.eosPosId);
  if ~isempty(posIds)
    srcPositions = tgtPos - (allSrcPos(posIds, tgtPos)' - params.zeroPosId); % src_pos = tgt_pos - relative_pos
    flags = (srcPositions<=0 | srcPositions>=srcLens(posIds)); % srcLen here include <eos> which we consider to be out of boundary
    outOfBoundaryIds = posIds(flags);
    posIds = setdiff(posIds, outOfBoundaryIds);
    srcPositions = srcPositions(~flags);
  else
    outOfBoundaryIds = [];
  end
  
  % null + out of boundary
  nullIds = [unmaskedIds(predPositions == params.nullPosId) outOfBoundaryIds];
  s_t(:, nullIds) = repmat(model.W_emb(:, params.nullPosId), 1, length(nullIds));

  % in boundary
  if ~isempty(posIds)
    % we had an extra <s_eos> on the src side. but srcLens doesn't count that additional <s_eos>
    colIndices = srcMaxLen-1-srcLens(posIds)+srcPositions;
    
    % use the below two lines to verify if you get the alignments correctly
    % params.vocab(input(sub2ind(size(input), posIds, colIndices)))
    % params.vocab(trainData.tgtOutput(posIds, tgtPos))
    
    if params.posModel==1 % use src embedding
      srcEmbIndices = input(sub2ind(size(input), posIds, colIndices));
      s_t(:, posIds) = model.W_emb(:, srcEmbIndices);
    elseif params.posModel==2 || params.posModel==3 % use src hidden states
      % topHidVecs: lstmSize * curBatchSize * T
      [linearIndices] = getTensorLinearIndices(trainData.srcHidVecs, posIds, colIndices);
      s_t(:, posIds) = reshape(trainData.srcHidVecs(linearIndices), params.lstmSize, length(posIds)); 
    end
  else
    colIndices = [];
    srcEmbIndices = [];
  end
  
  if params.posModel==1 % use src embeddings
    embIndices = zeros(1, params.curBatchSize);
    embIndices(posIds) = srcEmbIndices;
    embIndices(nullIds) = params.nullPosId;
    embIndices(eosIds) = params.eosPosId;
    embIndices = embIndices(unmaskedIds);
  else
    embIndices = [];
  end
  
  % store in structure
  srcPosData.eosIds = eosIds;
  srcPosData.nullIds = nullIds;
  srcPosData.posIds = posIds;
  srcPosData.colIndices = colIndices;
  srcPosData.embIndices = embIndices;
  
  % assert
  if params.assert
    assert(isempty(find(colIndices>=srcMaxLen, 1)));
    assert(isempty(setdiff(unmaskedIds, [posIds, nullIds, eosIds])));
    assert(length(unmaskedIds) == length([posIds, nullIds, eosIds]));
    assert(sum(sum(s_t(:, curMask.maskedIds)))==0);
    assert(sum(embIndices == (params.srcEos+params.tgtVocabSize))==0);
  end  
end

