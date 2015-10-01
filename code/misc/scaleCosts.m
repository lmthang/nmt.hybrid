function [curCosts] = scaleCosts(curCosts, counts)
  curCosts.total = curCosts.total/counts.total;
  curCosts.word = curCosts.word/counts.word;
end