function [curCosts] = scaleCosts(curCosts, counts, params)
  curCosts.total = curCosts.total/counts.total;
  curCosts.word = curCosts.word/counts.word;
end