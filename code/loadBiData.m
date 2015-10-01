function [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab, varargin) % , isDecode
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
    [srcSents] = loadMonoData(srcFile, chunkSize, params.baseIndex, srcVocab, 'src'); % , params.srcEos
  else
    srcSents = {};
  end
  
  % tgt
  tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
  [tgtSents, numSents] = loadMonoData(tgtFile, chunkSize, params.baseIndex, tgtVocab, 'tgt'); % , params.tgtEos
end

function [sents, numSents] = loadMonoData(file, numSents, baseIndex, vocab, label) % eos, 
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents); %, eos);
  fclose(fid);
  printSent(2, sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(2, sents{end}, vocab, ['  ', label, ' end:']);
end
