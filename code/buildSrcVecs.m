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
function [srcVecsSub, h2sInfo] = buildSrcVecs(srcVecsAll, srcPositions, curMask, params, h2sInfo) % startAttnIds, endAttnIds, startIds, endIds, indices
  posWin = params.posWin;
  numPositions = 2*posWin+1;
  [lstmSize, batchSize, numSrcHidVecs] = size(srcVecsAll);
  
  % masking
  srcPositions(curMask.maskedIds) = [];
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
  startAttnIds = srcPositions-posWin;
  indicesAll = reshape(bsxfun(@plus, startAttnIds(:), 0:(numPositions-1)), 1, []); 
  
  % check those that are out of boundaries
  excludeIds = find(indicesAll>numSrcHidVecs | indicesAll<1);
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
end

%% old version %%
%   posWin = params.posWin;
%   numPositions = 2*posWin+1;
%   [lstmSize, batchSize, numSrcHidVecs] = size(srcVecsAll);
%   srcVecs = zeroMatrix([lstmSize, batchSize*numPositions], params.isGPU, params.dataType);
%   
%   % these variables access srcVecsAll, lstmSize * batchSize * numSrcHidVecs
%   % telling us where to pay our attention to.
%   startAttnIds = srcPositions-posWin;
%   endAttnIds = srcPositions + posWin;
%   
%   % these variables are for srcVecs, lstmSize * batchSize * numPositions
%   % numPositions = 2*posWin+1
%   startIds = ones(1, batchSize);
%   endIds = numPositions*ones(1, batchSize);
%   
%   %% boundary condition for startAttnIds
%   indices = find(startAttnIds<1);
%   startIds(indices) = startIds(indices) - (startAttnIds(indices)-1);
%   startAttnIds(indices) = 1; % Note: don't swap these two lines
%   % here, we are sure that startHidId>=1, startAttnId>=1
%   
%   %% boundary condition for endAttnIds
%   indices = find(endAttnIds>numSrcHidVecs);
%   endIds(indices) = endIds(indices) - (endAttnIds(indices)-numSrcHidVecs);
%   endAttnIds(indices) = numSrcHidVecs; % Note: don't swap these two lines
%   % here, we are sure that endHidId<=numPositions, endAttnId<=numSrcHidVecs
%   
%   %% last boundary condition checks
%   flags = startIds<=endIds & startAttnIds<=endAttnIds & flags;
%   % out of boundary
%   indices = find(~flags);
%   startIds(indices) = 1; endIds(indices) = 0; startAttnIds(indices) = 1; endAttnIds(indices) = 0;
%   % in boundary
%   indices = find(flags);
%   if length(srcPositions)==1 && ~isempty(indices) % special case, when all src positions are the same. srcPositions is a scalar.
%     srcVecs(:, :, startIds:endIds) = srcVecsAll(:, :, startAttnIds:endAttnIds);
%   else % different source positions
%     for ii=1:length(indices)
%       index = indices(ii);
%       srcVecs(:, index, startIds(index):endIds(index)) = srcVecsAll(:, index, startAttnIds(index):endAttnIds(index));
%     end
%   end
