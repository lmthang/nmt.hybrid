function [srcHidVecs, linearIndices, unmaskedIds, attnLinearIndices] = buildSrcPosVecs(tgtPos, params, trainData, predPositions, curMask)
%%%
%
% For positional models, generate src vectors based on the predicted positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  %unmaskedIds = curMask.unmaskedIds;

  srcMaxLen = trainData.srcMaxLen; 
  srcLens = trainData.srcLens; %(unmaskedIds);
  
  %predPositions = predPositions(unmaskedIds);
  
  % exclude eos and null
  excludeFlags = predPositions==params.nullPosId | ~curMask.mask; % | predPositions==params.tgtEos | trainData.nullFlags; % TOFIX THIS LINE: no eos, double null check
  excludeIds = find(excludeFlags); %  | trainData.nullFlags
  unmaskedIds = find(~excludeFlags);
  if ~isempty(excludeIds)
    predPositions(excludeIds) = []; srcLens(excludeIds) = [];
  end
  
  %% compute aligned src positions
  if params.attnRelativePos
    srcPositions = tgtPos - (predPositions - params.zeroPosId); % src_pos = tgt_pos - relative_pos
  else % absolute position
    srcPositions = predPositions - params.zeroPosId;
  end
  
  % exclude those that are greater than params.maxSentLen
  excludeIds = find(srcPositions>params.maxSentLen);
  if ~isempty(excludeIds)
    srcPositions(excludeIds) = []; unmaskedIds(excludeIds) = []; srcLens(excludeIds) = [];
  end
  
  % cross right boundary
  indices = find(srcPositions>=srcLens); % srcLen here include <eos> which we consider to be out of boundary
  srcPositions(indices) = srcLens(indices)-1;
    
  if params.assert && params.isGradCheck==0
    assert(params.isReverse==1);
    assert(isempty(find(srcPositions<=0, 1)));
    %assert(isempty(find(srcPositions>=srcLens, 1)));
  elseif params.isGradCheck
    % TODO generate data with valid positions for grad check
    % cross left boundary
    srcPositions(srcPositions<=0) = 1; 
  end
  
  
  % get the column indices on the src side
  colIndices = srcMaxLen-srcPositions; % NOTE: IMPORTANT, here we assume src sentences are reversed

  % use the below two lines to verify if you get the alignments correctly
  %params.srcVocab(trainData.input(sub2ind(size(trainData.input), unmaskedIds, colIndices)))
  %params.tgtVocab(trainData.tgtOutput(unmaskedIds, tgtPos))

  %% get srcPosVecs
  % topHidVecs: lstmSize * curBatchSize * T
  if params.numAttnPositions>1
    % Here, colIndices derived below are for trainData.srcHidVecs, attnIndices are for srcHidVecs below
    % trainData.srcHidVecs: lstmSize * curBatchSize * srcMaxLen
    % srcHidVecs: lstmSize * curBatchSize * numAttnPositions
    % IMPORTANT: this line needs to go first before unmaskedIds is altered.
    attnIndices = reshape(repmat((1:params.numAttnPositions)', 1, length(unmaskedIds)), 1, []);
    
    unmaskedIds = reshape(repmat(unmaskedIds, params.numAttnPositions, 1), 1, []);
    startIds = colIndices-params.posWin;
    
    % Note: generate multiple sequences of the same lengths without using for loop, see this post for many elegant solutions
    % http://www.mathworks.com/matlabcentral/answers/217205-fast-ways-to-generate-multiple-sequences-without-using-for-loop
    % The below version is the only solution that is faster than for loop (3 times).
    colIndices = reshape(bsxfun(@plus, startIds(:), 0:(params.numAttnPositions-1))', 1, []);  
    
    % check those that are out of boundaries
    excludeIds = find(colIndices>=srcMaxLen | colIndices<1);
    if ~isempty(excludeIds)
      colIndices(excludeIds) = []; unmaskedIds(excludeIds) = []; attnIndices(excludeIds) = [];
    end
  else
    attnIndices = ones(1, length(unmaskedIds));
  end
  [linearIndices] = getTensorLinearIndices(trainData.srcHidVecs, unmaskedIds, colIndices);
  srcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
  
  [attnLinearIndices] = getTensorLinearIndices(srcHidVecs, unmaskedIds, attnIndices);
  srcHidVecs(attnLinearIndices) = trainData.srcHidVecs(linearIndices);
  %srcHidVecs(:, unmaskedIds) = reshape(trainData.srcHidVecs(linearIndices), params.lstmSize, length(unmaskedIds)); 
  
  % assert
  if params.assert
    assert(isempty(find(colIndices>=srcMaxLen, 1)));
    assert(sum(sum(srcHidVecs(:, curMask.maskedIds)))==0);
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
  