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
function [srcVecs, startAttnIds, endAttnIds, startIds, endIds, indices] = buildSrcVecs(srcVecsAll, srcPositions, flags, params)
  batchSize = size(srcVecsAll, 2);
  srcVecs = zeroMatrix([params.lstmSize, batchSize, params.numAttnPositions], params.isGPU, params.dataType);
  
  % these variables access srcVecsAll, lstmSize * batchSize * numSrcHidVecs
  % telling us where to pay our attention to.
  startAttnIds = srcPositions-params.posWin;
  endAttnIds = srcPositions + params.posWin;

  % these variables are for srcVecs, lstmSize * batchSize * numAttnPositions
  % numAttnPositions = 2*posWin+1
  startIds = ones(1, batchSize);
  endIds = params.numAttnPositions*ones(1, batchSize);
  
  %% boundary condition for startAttnIds
  indices = find(startAttnIds<1);
  startIds(indices) = startIds(indices) - (startAttnIds(indices)-1);
  startAttnIds(indices) = 1; % Note: don't swap these two lines
  % here, we are sure that startHidId>=1, startAttnId>=1
  
  %% boundary condition for endAttnIds
  indices = find(endAttnIds>params.numSrcHidVecs);
  endIds(indices) = endIds(indices) - (endAttnIds(indices)-params.numSrcHidVecs);
  endAttnIds(indices) = params.numSrcHidVecs; % Note: don't swap these two lines
  % here, we are sure that endHidId<=params.numAttnPositions, endAttnId<=params.numSrcHidVecs
  
  %% last boundary condition checks
  flags = startIds<=endIds & startAttnIds<=endAttnIds & flags;
  % out of boundary
  indices = find(~flags);
  startIds(indices) = 1; endIds(indices) = 0; startAttnIds(indices) = 1; endAttnIds(indices) = 0;
  % in boundary
  indices = find(flags);
  if length(srcPositions)==1 && ~isempty(indices) % special case, when all src positions are the same. srcPositions is a scalar.
    srcVecs(:, :, startIds:endIds) = srcVecsAll(:, :, startAttnIds:endAttnIds);
  else % different source positions
    for ii=1:length(indices)
      index = indices(ii);
      srcVecs(:, index, startIds(index):endIds(index)) = srcVecsAll(:, index, startAttnIds(index):endAttnIds(index));
    end
  end
end