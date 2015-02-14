function [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab)
  % src
  if params.isBi
    if params.isReverse
      srcFile = sprintf('%s.reversed.%s', prefix, params.srcLang);
    else
      srcFile = sprintf('%s.%s', prefix, params.srcLang);
    end
    [srcSents] = loadMonoData(srcFile, params.srcEos, -1, params.baseIndex, srcVocab, 'src');
  else
    srcSents = {};
  end
  
  % tgt
  tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
  [tgtSents, numSents] = loadMonoData(tgtFile, params.tgtEos, -1, params.baseIndex, tgtVocab, 'tgt');

  % prepare
  %[data] = prepareData(srcSents, tgtSents, params);
  %fprintf(2, '  numWords=%d\n', data.numWords);
end

function [sents, numSents] = loadMonoData(file, eos, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents, eos);
  fclose(fid);
  printSent(2, sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(2, sents{end}, vocab, ['  ', label, ' end:']);
end
