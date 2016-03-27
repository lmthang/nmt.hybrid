function [sents, numSents] = loadMonoData(file, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents);
  fclose(fid);
  printSent(2, sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(2, sents{end}, vocab, ['  ', label, ' end:']);
end