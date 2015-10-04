function [wordCounts] = updateCounts(wordCounts, data)
  wordCounts.total = wordCounts.total + data.numWords;
  wordCounts.word = wordCounts.word + data.numWords;
end