function [costs] = initCosts(params)
  costs.total = 0;
  costs.word = 0;
  if params.posSignal % positions
    costs.pos = 0;
    
    if params.predictNull
      costs.null = 0;
    end
  end
end