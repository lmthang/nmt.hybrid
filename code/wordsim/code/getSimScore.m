function score = getSimScore(word1, word2, We, vocabMap, distType, unkStr)
%%
  % word 1
  if isKey(vocabMap, word1)
    idx1 = vocabMap(word1);
  else
    idx1 = vocabMap(unkStr);
  end

  % word 2
  if isKey(vocabMap, word2)
    idx2 = vocabMap(word2);
  else
    idx2 = vocabMap(unkStr);
  end

  word1vec = We(:, idx1); 
  word2vec = We(:, idx2); 
  score = slmetric_pw(word1vec, word2vec, distType);
end
