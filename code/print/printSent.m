function printSent(fid, sent, vocab, prefix)
  fprintf(fid, '%s', prefix);
  for ii=1:length(sent)
    if ii<length(sent)
      fprintf(fid, '%s ', vocab{sent(ii)}); 
    else
      fprintf(fid, '%s\n', vocab{sent(ii)}); 
    end
  end
end


