function [curCosts] = updateCosts(curCosts, costs, params)
  curCosts.total = curCosts.total + costs.total;
  curCosts.word = curCosts.word + costs.word;
end
