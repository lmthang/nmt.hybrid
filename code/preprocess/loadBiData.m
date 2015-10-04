function [srcSents, tgtSents, numSents] = loadBiData(params, prefix, srcVocab, tgtVocab, varargin)
  chunkSize = -1;
  hasTgt = 1; % 1 -- has tgt file
  
  if length(varargin) >= 1
    chunkSize = varargin{1};
  end
  if length(varargin) >= 2
    hasTgt = varargin{2};
  end
  assert(params.isBi || hasTgt==1);
  
  % src
  if params.isBi
    if params.isReverse
      srcFile = sprintf('%s.%s.reversed', prefix, params.srcLang);
    else
      srcFile = sprintf('%s.%s', prefix, params.srcLang);
    end
    [srcSents, numSents] = loadMonoData(srcFile, chunkSize, params.baseIndex, srcVocab, 'src');
  else
    srcSents = {};
  end
  
  % tgt
  if hasTgt
    tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
    [tgtSents, numSents] = loadMonoData(tgtFile, chunkSize, params.baseIndex, tgtVocab, 'tgt');
  else
    tgtSents = repmat({[]}, 1, numSents);
  end
end

function [sents, numSents] = loadMonoData(file, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents);
  fclose(fid);
  printSent(2, sents{1}, vocab, ['  ', label, ' 1:']);
  printSent(2, sents{end}, vocab, ['  ', label, ' end:']);
end
