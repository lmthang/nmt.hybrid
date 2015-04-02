function [curSrcHidVecs, startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcHidVecs, srcMaxLen, tgtPos, params)

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

  curSrcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
  curSrcHidVecs(:, :, startHidId:endHidId) = srcHidVecs(:, :, startAttnId:endAttnId);

  % zero out the rest
  curSrcHidVecs(:, :, 1:startHidId-1) = 0;
  curSrcHidVecs(:, :, endHidId+1:end) = 0;
end