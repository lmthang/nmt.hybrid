function [srcPosVecs, linearIndices] = buildSrcPosVecs(tgtPos, params, trainData, predPositions, curMask)
%%%
%
% For positional models, generate src vectors based on the predicted positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  unmaskedIds = curMask.unmaskedIds;

  srcMaxLen = trainData.srcMaxLen; 
  srcLens = trainData.srcLens(unmaskedIds);
  
  predPositions = predPositions(unmaskedIds);
  
 
  %% compute aligned src positions
  srcPositions = tgtPos - (predPositions - params.zeroPosId); % src_pos = tgt_pos - relative_pos
  % cross left boundary
  srcPositions(srcPositions<=0) = 1; 
  % cross right boundary
  indices = find(srcPositions>=srcLens); % srcLen here include <eos> which we consider to be out of boundary
  srcPositions(indices) = srcLens(indices)-1;
  
  % get the column indices on the src side
  colIndices = srcMaxLen-srcPositions; % NOTE: important, here we assume src sentences are reversed

%   % use the below two lines to verify if you get the alignments correctly
%   params.vocab(input(sub2ind(size(input), unmaskedIds, colIndices)))
%   params.vocab(trainData.tgtOutput(unmaskedIds, tgtPos+1))

  %% get srcPosVecs
  % topHidVecs: lstmSize * curBatchSize * T
  [linearIndices] = getTensorLinearIndices(trainData.srcHidVecs, unmaskedIds, colIndices);
  srcPosVecs = zeroMatrix([params.lstmSize, params.curBatchSize], params.isGPU, params.dataType);
  srcPosVecs(:, unmaskedIds) = reshape(trainData.srcHidVecs(linearIndices), params.lstmSize, length(unmaskedIds)); 
  

  % assert
  if params.assert
    assert(isempty(find(colIndices>=srcMaxLen, 1)));
    assert(sum(sum(srcPosVecs(:, curMask.maskedIds)))==0);
  end  
end

%   if params.isReverse
%   else
%     colIndices = srcMaxLen-srcLens+srcPositions; % srcLens include <eos>
%   end

%assert(sum(srcEmbIndices == params.srcEos)==0);

%   if params.posModel==2 % use src embedding
%     srcEmbIndices = input(sub2ind(size(input), unmaskedIds, colIndices));
%     srcPosVecs(:, unmaskedIds) = model.W_emb(:, srcEmbIndices);
%     linearIndices = [];
%   elseif params.posModel==3 % use src hidden states % params.posModel==2 || 
%     srcEmbIndices = [];
%   end


  % store in structure
%   srcPosData.posIds = posIds;
%   srcPosData.colIndices = colIndices;
%   srcPosData.embIndices = embIndices;
%   srcPosData.srcPosVecs = srcPosVecs;
  