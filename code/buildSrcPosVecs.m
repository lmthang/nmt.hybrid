function [s_t, posIds, nullIds, eosIds, embIndices] = buildSrcPosVecs(t, model, params, trainData, curMask)
%%%
%
% For positional models, generate src vectors based on the predicted positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%

  srcMaxLen = trainData.srcMaxLen;
  unmaskedIds = curMask.unmaskedIds;
  srcPos = trainData.srcPos;
  srcLens = trainData.srcLens;
  input = trainData.input;
  srcEmbIndices = [];
  
  
  s_t = zeroMatrix([params.lstmSize, trainData.curBatchSize], params.isGPU, params.dataType);
  tgtPos = t-srcMaxLen+1;
  prevPositions = srcPos(unmaskedIds, tgtPos)';

  % eos
  eosIds = unmaskedIds(prevPositions == params.eosPosId);
  s_t(:, eosIds) = repmat(model.W_emb(:, params.eosPosId), 1, length(eosIds));

  % pos
  posIds = unmaskedIds(prevPositions ~= params.nullPosId & prevPositions ~= params.eosPosId); %setdiff(unmaskedIds, [eosIds nullIds]);
  if ~isempty(posIds)
    srcPositions = tgtPos - (srcPos(posIds, t-srcMaxLen+1)' - params.zeroPosId); % src_pos = tgt_pos - relative_pos
    flags = (srcPositions<=0 | srcPositions>=srcLens(posIds)); % srcLen here include <eos> which we consider to be out of boundary
    outOfBoundaryIds = posIds(flags);
    posIds = setdiff(posIds, outOfBoundaryIds);
    srcPositions = srcPositions(~flags);
  else
    outOfBoundaryIds = [];
  end

  % null + out of boundary
  nullIds = [unmaskedIds(prevPositions == params.nullPosId) outOfBoundaryIds];
  s_t(:, nullIds) = repmat(model.W_emb(:, params.nullPosId), 1, length(nullIds));

  % in boundary
  if ~isempty(posIds)
    colIndices = srcMaxLen-srcLens(posIds)+srcPositions;

    if params.posModel==1 % use src embedding
      srcEmbIndices = input(sub2ind(size(input), posIds, colIndices));
      s_t(:, posIds) = model.W_emb(:, srcEmbIndices);
    end
    
    % assert
    if params.assert
      assert(isempty(find(colIndices>=srcMaxLen, 1)));
      assert(isempty(setdiff(unmaskedIds, [posIds, nullIds, eosIds])));
    end
  end
  
  if params.posModel==1 % use src embedding
    embIndices = zeros(1, trainData.curBatchSize);
    embIndices(posIds) = srcEmbIndices;
    embIndices(nullIds) = params.nullPosId;
    embIndices(eosIds) = params.eosPosId;
    embIndices = embIndices(unmaskedIds);
  end
end