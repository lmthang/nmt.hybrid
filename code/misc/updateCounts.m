function [wordCounts] = updateCounts(wordCounts, data, params)
  wordCounts.total = wordCounts.total + data.numWords;
  wordCounts.word = wordCounts.word + data.numWords;
  if params.predictPos
    wordCounts.pos = wordCounts.pos + data.numPositions;
    if params.predictNull
      wordCounts.null = wordCounts.null + data.numNulls;
    end
  end
end