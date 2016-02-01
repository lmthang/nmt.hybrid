function [costs] = initCosts(params)
  costs.word = 0;
  if params.charOpt
    costs.char = 0;
  end
end