function printSentPos(fid, sent, vocab, prefix)
  fprintf(fid, '%s', prefix);
  for ii=1:length(sent)
    if ii<length(sent)
      if mod(ii, 2)==0 % word
        fprintf(fid, '%s ', vocab{sent(ii)}); 
      else % pos
        fprintf(fid, '%d ', sent(ii)); 
      end
    else % last word
      fprintf(fid, '%s\n', vocab{sent(ii)}); 
    end
  end
end


