function scores = getSimScores(wordPairs, We, vocabMap, distType, unkStr)
%%
  scores = zeros(length(wordPairs),1);

  for i = 1:length(wordPairs)
    word1 = wordPairs{i,1};
    word2 = wordPairs{i,2};
    %fprintf(1, '%s\t%s\n', word1, word2);
    scores(i) = getSimScore(word1, word2, We, vocabMap, distType, unkStr);
  end

  if strcmp(distType, 'corrdist')
    scores = 2-scores;
  else
    scores = -1*scores;
  end
end
