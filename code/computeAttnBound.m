function [startIds, endIds, startAttnIds, endAttnIds] = computeAttnBound(srcPositions, params)
  batchSize = length(srcPositions);
  
  % where to pay attention to on the source side (params.numSrcHidVecs)
  startAttnIds = srcPositions-params.posWin;
  endAttnIds = srcPositions + params.posWin;
  
  % where to get the align weights (numAttnPositions = 2*params.posWin+1)
  startIds = oneMatrix([1, batchSize], params.isGPU, params.dataType);
  endIds = params.numAttnPositions*startIds;
  
  %% boundary condition for startAttnIds
  indices = find(startAttnIds<1);
  startIds(indices) = startIds(indices) - (startAttnIds(indices)-1);
  startAttnIds(indices) = 1; % Note: don't swap these two lines
  % here, we are sure that startId>=1, startAttnId>=1
  
  %% boundary condition for endAttnIds
  indices = find(endAttnIds>params.numSrcHidVecs);
  endIds(indices) = endIds(indices) - (endAttnIds(indices)-params.numSrcHidVecs);
  endAttnIds(indices) = params.numSrcHidVecs; % Note: don't swap these two lines
  % here, we are sure that endId<=numAttnPositions, endAttnId<=params.numSrcHidVecs
  
  %% last boundary condition checks
  flags = startIds<=endIds & startAttnIds<=endAttnIds; % & flags;
  % out of boundary
  indices = find(~flags);
  startIds(indices) = 1; endIds(indices) = 0; startAttnIds(indices) = 1; endAttnIds(indices) = 0;
end
