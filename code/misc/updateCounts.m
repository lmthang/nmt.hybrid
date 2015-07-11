function [wordCounts] = updateCounts(wordCounts, data, params)
  wordCounts.total = wordCounts.total + data.numWords;
  wordCounts.word = wordCounts.word + data.numWords;
end