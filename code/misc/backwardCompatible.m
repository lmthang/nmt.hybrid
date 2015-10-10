function [params] = backwardCompatible(params, fieldNames)
  for ii=1:length(fieldNames)
    field = fieldNames{ii};
    if ~isfield(params, field)
      params.(field) = 0;
    end
  end
end