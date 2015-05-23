function [posMask] = createPosMask(tgtPos, params, trainData, curMask)
  if params.predictPos
    curPosOutput = trainData.posOutput(:, tgtPos)';
    if params.predictPos==1
      posMask.mask = curMask.mask & curPosOutput~=params.nullPosId;
    elseif params.predictPos==2
      posMask.mask = curMask.mask & curPosOutput~=params.nullPosId & curPosOutput~=params.tgtEos;
    end
  else
    posMask.mask = curMask.mask;
  end
  posMask.unmaskedIds = find(posMask.mask);
  posMask.maskedIds = find(~posMask.mask);
end