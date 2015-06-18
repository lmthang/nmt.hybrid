function [curCosts] = scaleCosts(curCosts, counts, params)
  curCosts.total = curCosts.total/counts.total;
  curCosts.word = curCosts.word/counts.word;

  if params.posSignal
    curCosts.pos = curCosts.pos/counts.pos;
  end
end

%     if params.predictNull
%       curCosts.null = curCosts.null/counts.null;
%     end