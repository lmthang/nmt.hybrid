%% Prepare data
%
%  Thang Luong @ 2014, 2015, <lmthang@stanford.edu>
%
%  organize data into matrix format and produce masks, add tgtVocabSize to srcSents:
%   input:      numSents * (srcMaxLen+tgtMaxLen-1)
%   tgtOutput: numSents * tgtMaxLen
%   tgtMask  : numSents * tgtMaxLen, indicate where to ignore in the tgtOutput
%  For the monolingual case, each src sent contains a single simple tgtSos,
%   hence srcMaxLen = 1
function [data] = prepareData(srcSents, tgtSents, isTest, params, varargin)
  if length(varargin)==2
    srcLens = varargin{1};
    tgtLens = varargin{2};
  else
    tgtLens = cellfun(@(x) length(x), tgtSents);
    srcLens = cellfun(@(x) length(x), srcSents);
  end
  
  numSents = length(tgtSents);
  if params.isBi
    srcZeroId = params.srcSosVocabId;
    
    if isTest==0 || params.attnFunc==1 % limit sent lengths for training or for attention model during both training/testing
      srcLens(srcLens>params.maxSentLen) = params.maxSentLen; 
      srcMaxLen = params.maxSentLen;
    else
      srcMaxLen = max(srcLens);
    end
    
    if params.posModel>0 % add an extra <s_eos> to the src side
      srcMaxLen = srcMaxLen+1;
    end
  else
    srcLens = ones(numSents, 1);
    srcZeroId = params.tgtSos;
    srcMaxLen = 1;
  end
  
  % positional models, tgt sent: pos1 word1 ... pos_n word_n <eos>
  if params.posModel>0
    tgtLens = (tgtLens+1)/2;
  end
  if isTest==0
    tgtLens(tgtLens>params.maxSentLen) = params.maxSentLen; % limit sent lengths
    tgtMaxLen = max(tgtLens);
    assert(tgtMaxLen<=params.maxSentLen);
  else
    tgtMaxLen = max(tgtLens);
  end

  
  %% input / output
  input = [srcZeroId*ones(numSents, srcMaxLen) params.tgtEos*ones(numSents, tgtMaxLen-1)]; % size numSents * (srcMaxLen + tgtMaxLen - 1)
  tgtOutput = params.tgtEos*ones(numSents, tgtMaxLen);
  
  % positional models
  if params.posModel>0
    srcPos = params.eosPosId*ones(numSents, tgtMaxLen); % since tgt sent: pos1 word1 ... pos_n word_n <eos>. Later we want: pos1 ... pos_n pos_eos.
  end
  
  for ii=1:numSents
    %% src
    if params.isBi
      srcLen = srcLens(ii);
      if params.posModel>0 % add an extra <s_eos> to the src side
        input(ii, srcMaxLen-srcLen:srcMaxLen-1) = srcSents{ii}(1:srcLen) + params.tgtVocabSize; % src part
        input(ii, srcMaxLen) = params.srcEosVocabId;
      else
        input(ii, srcMaxLen-srcLen+1:srcMaxLen) = srcSents{ii}(1:srcLen) + params.tgtVocabSize; % src part
      end
    end
    
    %% tgt
    tgtSent = tgtSents{ii};
    tgtLen = tgtLens(ii);
    
    % positional models
    if params.posModel>0
      % words
      tgtSent(1:2:2*tgtLen-2) = []; % remove positions
      
      % positions
      srcPos(ii, 1:tgtLen-1) = tgtSents{ii}(1:2:2*tgtLen-2); % positions
    end
    
    input(ii, srcMaxLen+1:srcMaxLen+tgtLen-1) = tgtSent(1:tgtLen-1); % tgt part
    tgtOutput(ii, 1:tgtLen) = tgtSent(1:tgtLen);
  end
  
  if params.isBi
    inputMask = (input~=srcZeroId & input~=params.tgtEos);
  else % for mono case, we still learn parameters for the srcZeroId which is tgtSos.
    inputMask = (input~=params.tgtEos);
  end
  numWords = sum(sum(inputMask(:, srcMaxLen:end))); 
  
  % sanity check
  if params.assert
    % the last src symbol needs to be eos for all sentences
    if params.isBi
      assert(length(unique(input(:, srcMaxLen)))==1); 
      srcEos = srcSents{1}(end) + params.tgtVocabSize;
      assert(input(1, srcMaxLen)==srcEos);
    end
    
    assert(numWords == sum(tgtLens));
  end
  
  data.input = input;
  data.inputMask = inputMask;
  data.tgtOutput = tgtOutput;
  data.srcMaxLen = srcMaxLen;
  data.tgtMaxLen = tgtMaxLen;
  data.numWords = numWords;
  data.srcLens = srcLens;
  
  % positional models
  if params.posModel>0
    data.srcPos = srcPos;
  end
end

    %label = 'input';
    %printSent(input(1, :), params.vocab, ['  ', label, ' 1:']);
    %printSent(input(end, :), params.vocab, ['  ', label, ' end:']);
    
