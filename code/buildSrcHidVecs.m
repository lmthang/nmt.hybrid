function [srcHidVecs, startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcHidVecsAll, srcMaxLen, tgtPos, params)

  startAttnId = (srcMaxLen-tgtPos)-params.posWin;
  endAttnId = (srcMaxLen-tgtPos) + params.posWin;

  startHidId = 1;
  endHidId = params.numAttnPositions;
  if startAttnId<1
    startHidId = startHidId - (startAttnId-1);
    startAttnId = 1; % Note: don't swap these two lines
  end
  if endAttnId>params.numSrcHidVecs
    endHidId = endHidId - (endAttnId-params.numSrcHidVecs);
    endAttnId = params.numSrcHidVecs; % Note: don't swap these two lines
  end
  
  batchSize = size(srcHidVecsAll, 2);
  srcHidVecs = zeroMatrix([params.lstmSize, batchSize, params.numAttnPositions], params.isGPU, params.dataType);
  if startHidId<=params.numAttnPositions && endHidId>=1 && startHidId<=endHidId && startAttnId<=endAttnId % in boundary
    srcHidVecs(:, :, startHidId:endHidId) = srcHidVecsAll(:, :, startAttnId:endAttnId);
  else % out of boundary
    startHidId = 1;
    endHidId = 0;
    startAttnId = 1;
    endAttnId = 0;
  end
end