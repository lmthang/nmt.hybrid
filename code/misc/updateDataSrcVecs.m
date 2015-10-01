function [data] = updateDataSrcVecs(data, params)
  if params.attnGlobal % global
    data.absSrcHidVecs = data.srcHidVecsOrig;
  else % local
    data.srcHidVecs = data.srcHidVecsOrig;
  end
end