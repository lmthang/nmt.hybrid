function [costStr] = getCostStr(costStruct)
  costStr = sprintf('%.2f (', costStruct.total);
  fields = {'pos', 'null', 'word'};
 
  count = 0;
  for ii=1:length(fields)
    field = fields{ii};
    if isfield(costStruct, field)
      costStr = sprintf('%s%.2f, ', costStr, costStruct.(field));
      count = count+1;
    end
  end
  
  costStr(end-1:end) = []; % remove the end part
  if count>0 % there're other fields
    costStr = sprintf('%s)', costStr);
  end
end