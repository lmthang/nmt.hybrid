function [wordCounts] = updateCounts(wordCounts, data)
  wordCounts.word = wordCounts.word + data.numWords;
  if isfield(data, 'numChars')
    wordCounts.char = wordCounts.char + data.numChars;
  end
end