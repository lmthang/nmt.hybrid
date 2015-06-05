function [curCosts] = updateCosts(curCosts, costs, params)
  curCosts.total = curCosts.total + costs.total;
  curCosts.word = curCosts.word + costs.word;

  if params.posSignal % positions
    curCosts.pos = curCosts.pos + costs.pos;    
  end
end

%     if params.predictNull
%       curCosts.null = curCosts.null + costs.null;
%     end
