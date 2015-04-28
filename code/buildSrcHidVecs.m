function [startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcMaxLen, tgtPos, params)

  startAttnId = (srcMaxLen-tgtPos)-params.posWin;
  endAttnId = (srcMaxLen-tgtPos) + params.posWin;

  startHidId = 1;
  endHidId = params.numAttnPositions;
  if startAttnId<1
    startHidId = startHidId - (startAttnId-1);
    if startHidId>params.numAttnPositions
      startHidId = params.numAttnPositions+1;
    end
    startAttnId = 1; % Note: don't swap these two lines
  end
  if endAttnId>params.numSrcHidVecs
    endHidId = endHidId - (endAttnId-params.numSrcHidVecs);
    if endHidId<0
      endHidId=0;
    end
    endAttnId = params.numSrcHidVecs; % Note: don't swap these two lines
  end
end