function [curCosts] = scaleCosts(curCosts, counts, params)
  curCosts.total = curCosts.total/counts.total;
  curCosts.word = curCosts.word/counts.word;

  if params.predictPos % positions
    curCosts.pos = curCosts.pos/counts.pos;
    
    if params.predictNull
      curCosts.null = curCosts.null/counts.null;
    end
  end
end