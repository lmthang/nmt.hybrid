function [startIds, endIds, startAttnIds, endAttnIds] = computeAttnBound(srcPositions, posWin, numAttnPositions, numSrcHidVecs)
  batchSize = length(srcPositions);
  
  % these variables access srcVecsAll, lstmSize * batchSize * numSrcHidVecs
  % telling us where to pay our attention to.
  startAttnIds = srcPositions-posWin;
  endAttnIds = srcPositions + posWin;
  
  % these variables are for srcVecs, lstmSize * batchSize * numPositions
  % numPositions = 2*posWin+1
  startIds = ones(1, batchSize);
  endIds = numAttnPositions*ones(1, batchSize);
  
  %% boundary condition for startAttnIds
  indices = find(startAttnIds<1);
  startIds(indices) = startIds(indices) - (startAttnIds(indices)-1);
  startAttnIds(indices) = 1; % Note: don't swap these two lines
  % here, we are sure that startHidId>=1, startAttnId>=1
  
  %% boundary condition for endAttnIds
  indices = find(endAttnIds>numSrcHidVecs);
  endIds(indices) = endIds(indices) - (endAttnIds(indices)-numSrcHidVecs);
  endAttnIds(indices) = numSrcHidVecs; % Note: don't swap these two lines
  % here, we are sure that endHidId<=numPositions, endAttnId<=numSrcHidVecs
  
  %% last boundary condition checks
  flags = startIds<=endIds & startAttnIds<=endAttnIds; % & flags;
  % out of boundary
  indices = find(~flags);
  startIds(indices) = 1; endIds(indices) = 0; startAttnIds(indices) = 1; endAttnIds(indices) = 0;
end

%   posWin = params.posWin;
%   numPositions = 2*posWin+1;
%   [lstmSize, batchSize, numSrcHidVecs] = size(srcVecsAll);
%   srcVecs = zeroMatrix([lstmSize, batchSize*numPositions], params.isGPU, params.dataType);
  
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
