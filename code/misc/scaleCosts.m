function [curCosts] = scaleCosts(curCosts, counts)
  fieldNames = fields(curCosts);
  for ii=1:length(fieldNames)
    field = fieldNames{ii}; 
    curCosts.(field) = curCosts.(field) / counts.(field);
  end
end