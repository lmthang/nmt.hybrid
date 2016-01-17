function [costs] = initCosts(params)
  costs.total = 0;
  if params.charOpt
    costs.word = 0;
    costs.char = 0;
  end
end