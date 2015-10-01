function [data] = updateDataSrcVecs(data, params)
  if params.attnGlobal % global
    if params.attnOpt==0 % fixed-length alignments
      data.absSrcHidVecs = zeroMatrix([params.lstmSize, params.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
      data.absSrcHidVecs(:, :, params.numAttnPositions-params.numSrcHidVecs+1:end) = data.srcHidVecsOrig;
    else % variable-length alignments
      data.absSrcHidVecs = data.srcHidVecsOrig;
    end
  else % local
    data.srcHidVecs = data.srcHidVecsOrig;
  end
end