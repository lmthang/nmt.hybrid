function [s_t, posIds, nullIds, eosIds, colIndices, embIndices] = buildSrcPosVecs(s_t, t, model, params, trainData, curMask)
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
  
  % zero out those that do not participate
  s_t(:, curMask.maskedIds) = 0;
  
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
    colIndices = srcMaxLen-srcLens(posIds)+srcPositions;

    if params.posModel==1 % use src embedding
      srcEmbIndices = input(sub2ind(size(input), posIds, colIndices));
      s_t(:, posIds) = model.W_emb(:, srcEmbIndices);
      % use the below two lines to verify if you get the alignments correctly
      % params.vocab(input(sub2ind(size(input), posIds, colIndices)))
      % params.vocab(trainData.tgtOutput(posIds, tgtPos))
    elseif params.posModel==2 % use src hidden states
      % srcHidVecs: lstmSize * curBatchSize * maxSentLen
      % posIds colIndices
      numPositions = length(posIds);
      xIds = repmat(1:params.lstmSize, 1, numPositions);
      yIds = repmat(posIds, params.lstmSize, 1);
      yIds = yIds(:)';
      zIds = repmat(colIndices, params.lstmSize, 1);
      zIds = zIds(:)';
      s_t(:, posIds) = reshape(trainData.srcHidVecs(sub2ind(size(trainData.srcHidVecs), xIds, yIds, zIds)), params.lstmSize, numPositions); 
    end
  else
    colIndices = [];
    srcEmbIndices = [];
  end
  
  if params.posModel==1 % use src embeddings
    embIndices = zeros(1, trainData.curBatchSize);
    embIndices(posIds) = srcEmbIndices;
    embIndices(nullIds) = params.nullPosId;
    embIndices(eosIds) = params.eosPosId;
    embIndices = embIndices(unmaskedIds);
  else
    embIndices = [];
  end
  
  % assert
  if params.assert
    assert(isempty(find(colIndices>=srcMaxLen, 1)));
    assert(isempty(setdiff(unmaskedIds, [posIds, nullIds, eosIds])));
    assert(length(unmaskedIds) == length([posIds, nullIds, eosIds]));
    assert(sum(sum(s_t(:, curMask.maskedIds)))==0);
    assert(sum(embIndices == (params.srcEos+params.tgtVocabSize))==0);
  end
end