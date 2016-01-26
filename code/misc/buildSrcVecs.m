%%%
%
% Gather src hidden vectors for attention-based models.
% For each sentence ii, we extract a set of vectors to pay attention to
%   srcVecsAll(:, ii, srcPositions(ii)-posWin:srcPositions(ii)+posWin)
%   and put it into srcVecs(:, ii, :). Boundary cases are handled as well.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 

% TODO move srcMaxLen out
% IMPORTANT: we assume the sentences are reversed here
% srcPositions = srcMaxLen - srcPositions
function [srcVecsSub, h2sInfo] = buildSrcVecs(srcVecsAll, srcPositions, curMask, srcLens, srcMaxLen, params, h2sInfo)
  posWin = params.posWin;
  numPositions = 2*posWin+1;
  [lstmSize, batchSize, numSrcHidVecs] = size(srcVecsAll);
  
  % masking
  srcPositions(curMask.maskedIds) = [];
  srcLens(curMask.maskedIds) = [];
  unmaskedIds = curMask.unmaskedIds;
  
  % flatten matrices of size batchSize*numPositions (not exactly batch size but close)
  % init. IMPORTANT: don't swap these two lines
  % assume unmaskedIds = [1, 2, 3, 4, 5], numPositions=3
  indicesSub = reshape(repmat(1:numPositions, length(unmaskedIds), 1), 1, []); % 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3
  unmaskedIds = repmat(unmaskedIds, 1, numPositions); % 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5
  
  % Note: generate multiple sequences of the same lengths without using for loop, see this post for many elegant solutions
  % http://www.mathworks.com/matlabcentral/answers/217205-fast-ways-to-generate-multiple-sequences-without-using-for-loop
  % The below version is the only solution that is faster than for loop (3 times).
  % If startAttnIds = [ 2 4 6 8 10 ] and numPositions=3
  % then indicesAll = [ 2 4 6 8 10 3 5 7 9 11 4 6 8 10 12 ].
  srcLens = repmat(srcLens, [1, numPositions]);
  startAttnIds = srcPositions-posWin;
  indicesAll = reshape(bsxfun(@plus, startAttnIds(:), 0:(numPositions-1)), 1, []); 
  
  % check those that are out of boundaries
  excludeIds = find(indicesAll>numSrcHidVecs | indicesAll<(srcMaxLen-srcLens+1));
  if ~isempty(excludeIds)
    indicesAll(excludeIds) = []; unmaskedIds(excludeIds) = []; indicesSub(excludeIds) = [];
  end
  
  h2sInfo.indicesAll = indicesAll;
  h2sInfo.unmaskedIds = unmaskedIds;
  
  if ~isempty(unmaskedIds)
    srcVecsSub = zeroMatrix([lstmSize, batchSize*numPositions], params.isGPU, params.dataType);
    
    % create linear indices
    h2sInfo.linearIdSub = sub2ind([batchSize, numPositions], unmaskedIds, indicesSub);
    h2sInfo.linearIdAll = sub2ind([batchSize, numSrcHidVecs], unmaskedIds, indicesAll);

    % create srcVecs
    srcVecsAll = reshape(srcVecsAll, lstmSize, []);
    srcVecsSub(:, h2sInfo.linearIdSub) = srcVecsAll(:, h2sInfo.linearIdAll);
    srcVecsSub = reshape(srcVecsSub, [lstmSize, batchSize, numPositions]);
  else
    srcVecsSub = zeroMatrix([lstmSize, batchSize, numPositions], params.isGPU, params.dataType);
    h2sInfo.linearIdSub = [];
    h2sInfo.linearIdAll = [];
  end
  
  h2sInfo.alignMask = zeroMatrix([batchSize, numPositions], params.isGPU, params.dataType);
  h2sInfo.alignMask(h2sInfo.linearIdSub) = 1;
  h2sInfo.alignMask = h2sInfo.alignMask'; % numPositions * batchSize
  h2sInfo.srcPositions = srcPositions;
end
