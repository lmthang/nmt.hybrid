function [params] = backwardCompatible(params, fieldNames, defaultValue)
  for ii=1:length(fieldNames)
    field = fieldNames{ii};
    if ~isfield(params, field)
      params.(field) = defaultValue;
    end
  end
end