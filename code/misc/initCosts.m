function [costs] = initCosts(params)
  costs.word = 0;
  if params.charOpt > 1
    costs.char = 0;
  end
end