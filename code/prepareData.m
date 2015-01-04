
%% Prepare data
%
%  Thang Luong @ 2014, <lmthang@stanford.edu>
%
%  organize data into matrix format and produce masks, add tgtVocabSize to srcSents:
%   input:      numSents * (srcMaxLen+tgtMaxLen-1)
%   tgtOutput: numSents * tgtMaxLen
%   tgtMask  : numSents * tgtMaxLen, indicate where to ignore in the tgtOutput
%  For the monolingual case, each src sent contains a single simple tgtSos,
%   hence srcMaxLen = 1
function [input, inputMask, tgtOutput, srcMaxLen, tgtMaxLen, srcLens] = prepareData(srcSents, tgtSents, params)
  if params.isBi
    srcZeroId = params.tgtVocabSize + params.srcSos;
    srcMaxLen = max(cellfun(@(x) length(x), srcSents));
  else
    srcZeroId = params.tgtSos;
    srcMaxLen = 1;
  end
  numSents = length(tgtSents);
  tgtMaxLen = max(cellfun(@(x) length(x), tgtSents));
  input = [srcZeroId*ones(numSents, srcMaxLen) params.tgtEos*ones(numSents, tgtMaxLen-1)];
  tgtOutput = params.tgtEos*ones(numSents, tgtMaxLen);
  srcLens = zeros(numSents, 1);
  for ii=1:numSents
    if params.isBi
      srcLen = length(srcSents{ii});
      input(ii, srcMaxLen-srcLen+1:srcMaxLen) = srcSents{ii} + params.tgtVocabSize; % src part
      srcLens(ii) = srcLen;
    end
    
    tgtLen = length(tgtSents{ii});
    input(ii, srcMaxLen+1:srcMaxLen+tgtLen-1) = tgtSents{ii}(1:end-1); % tgt part
    tgtOutput(ii, 1:tgtLen) = tgtSents{ii};
  end
  
  %tgtMask = (tgtOutput~=params.tgtEos);
  if params.isBi
    inputMask = (input~=srcZeroId & input~=params.tgtEos);
  else % for mono case, we still learn parameters for the srcZeroId which is tgtSos.
    inputMask = (input~=params.tgtEos);
  end
  
  % the last src symbol needs to be eos for all sentences
  if params.isBi
    assert(length(unique(input(:, srcMaxLen)))==1); 
    srcEos = srcSents{1}(end) + params.tgtVocabSize;
    assert(input(1, srcMaxLen)==srcEos);
  end
end