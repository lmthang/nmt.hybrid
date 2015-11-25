function [toStruct] = copyStruct(fromStruct, toStruct)
  fieldList = fields(fromStruct);
  for ii=1:length(fieldList)
    field = fieldList{ii};
    toStruct.(field) = fromStruct.(field);
  end
end