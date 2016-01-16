function printSentChar(fid, sent, vocab, prefix, positions, rareWords)
  fprintf(fid, '%s', prefix);
  
  words = vocab(sent);
  words(positions) = rareWords;
  for ii=1:length(sent)
    if ii<length(sent)
      fprintf(fid, '%s ', words{ii}); 
    else
      fprintf(fid, '%s\n', words{ii}); 
    end
  end
end


