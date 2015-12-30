function [wordCounts] = updateCounts(wordCounts, data)
  wordCounts.total = wordCounts.total + data.numWords;
  wordCounts.word = wordCounts.word + data.numWords;
  if isfield(data, 'numChars')
    wordCounts.total = wordCounts.total + data.numChars;
    wordCounts.char = wordCounts.char + data.numChars;
  end
end