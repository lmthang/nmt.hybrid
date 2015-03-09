function [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab, isDecode, varargin)
  if length(varargin) == 1
    chunkSize = varargin{1};
  else
    chunkSize = -1;
  end
  
  % src
  if params.isBi
    if params.isReverse
      srcFile = sprintf('%s.reversed.%s', prefix, params.srcLang);
    else
      srcFile = sprintf('%s.%s', prefix, params.srcLang);
    end
    [srcSents] = loadMonoData(srcFile, params.srcEos, chunkSize, params.baseIndex, srcVocab, 'src');
  else
    srcSents = {};
  end
  
  % tgt
  if (isDecode==0)
    tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
    [tgtSents, numSents] = loadMonoData(tgtFile, params.tgtEos, chunkSize, params.baseIndex, tgtVocab, 'tgt');
  else
    tgtSents = {};
    numSents = 0;
  end
end

function [sents, numSents] = loadMonoData(file, eos, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents, eos);
  fclose(fid);
  printSent(2, sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(2, sents{end}, vocab, ['  ', label, ' end:']);
end
