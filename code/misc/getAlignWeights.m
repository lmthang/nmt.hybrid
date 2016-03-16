function [alignWeights, alignIndices] = getAlignWeights(attnInfos, srcLens, models, params)
  assert(params.attnFunc>0);
  batchSize = length(srcLens);
  numModels = length(models);

  % init
  alignWeights = cell(batchSize, 1);
  for sentId=1:batchSize % go through each sent
    alignWeights{sentId} = zeroMatrix([srcLens(sentId)-1, 1], params.isGPU, params.dataType); % ignore eos
  end

  % average alignment weights
  for mm=1:numModels
    if models{mm}.params.attnFunc==0 % non-attention
      continue;
    end

    if models{mm}.params.attnGlobal==0 % local
      [startIds, endIds, startAttnIds, endAttnIds] = computeAttnBound(attnInfos{mm}.srcPositions, models{mm}.params);
    end

    for sentId=1:batchSize % go through each sent
      srcLen = srcLens(sentId);

      if models{mm}.params.attnGlobal
        alignWeights{sentId} = alignWeights{sentId} + attnInfos{mm}.alignWeights(end-srcLen+2:end, sentId);
      else
        if startIds(sentId)<=endIds(sentId)
          offset = params.srcMaxLen-srcLen;

          % out of boundary
          if startAttnIds(sentId) <= offset
            startIds(sentId) = startIds(sentId) + offset + 1 - startAttnIds(sentId);
            startAttnIds(sentId) = offset + 1;
          end

          indices = startAttnIds(sentId)-offset:endAttnIds(sentId)-offset;
          alignWeights{sentId}(indices) = alignWeights{sentId}(indices) + attnInfos{mm}.alignWeights(startIds(sentId):endIds(sentId), sentId);
        end
      end
    end
  end

  % get alignment index
  alignIndices = zeroMatrix([1, batchSize], params.isGPU, params.dataType);
  for sentId=1:batchSize % go through each sent
    [~, alignIndices(sentId)] = max(alignWeights{sentId}, [], 1);
  end
end

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
