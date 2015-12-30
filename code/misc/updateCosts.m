function [curCosts] = updateCosts(curCosts, costs)
  fieldNames = fields(curCosts);
  for ii=1:length(fieldNames)
    field = fieldNames{ii}; 
    curCosts.(field) = curCosts.(field) + costs.(field);
  end
end
