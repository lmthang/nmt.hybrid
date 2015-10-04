function [curCosts] = updateCosts(curCosts, costs)
  curCosts.total = curCosts.total + costs.total;
  curCosts.word = curCosts.word + costs.word;
end
