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
