function [srcHidVecs, startAttnId, endAttnId, startHidId, endHidId] = buildSrcHidVecs(srcHidVecsAll, srcMaxLen, tgtPos, params)

  startAttnId = (srcMaxLen-tgtPos)-params.posWin;
  endAttnId = (srcMaxLen-tgtPos) + params.posWin;

  startHidId = 1;
  if startAttnId<1
    startHidId = startHidId - (startAttnId-1);
    startAttnId = 1; % Note: don't swap these two lines
  end
  % here, we are sure that startHidId>=1, startAttnId>=1
  
  endHidId = params.numAttnPositions;
  if endAttnId>params.numSrcHidVecs
    endHidId = endHidId - (endAttnId-params.numSrcHidVecs);
    endAttnId = params.numSrcHidVecs; % Note: don't swap these two lines
  end
  % here, we are sure that endHidId<=params.numAttnPositions, endAttnId<=params.numSrcHidVecs
  
  batchSize = size(srcHidVecsAll, 2);
  srcHidVecs = zeroMatrix([params.lstmSize, batchSize, params.numAttnPositions], params.isGPU, params.dataType);
  
  if startHidId<=endHidId && startAttnId<=endAttnId % in boundary
    srcHidVecs(:, :, startHidId:endHidId) = srcHidVecsAll(:, :, startAttnId:endAttnId);
  else % out of boundary
    startHidId = 1;
    endHidId = 0;
    startAttnId = 1;
    endAttnId = 0;
  end
end

%     if startHidId<=0 || startHidId>size(srcHidVecs, 3) || endHidId<=0 || endHidId>size(srcHidVecs, 3) ...
%         || startAttnId<=0 || startAttnId>size(srcHidVecsAll,3)|| endAttnId<=0 || endAttnId>size(srcHidVecsAll,3)
%       size(srcHidVecs)
%       size(srcHidVecsAll)
%     end