function [data] = prepareData(srcSents, tgtSents, isTest, params, varargin)
%  Organize data into matrix format and produce masks, add tgtVocabSize to srcSents:
%   input:      numSents * (srcMaxLen+tgtMaxLen-1)
%   tgtOutput: numSents * tgtMaxLen
%   tgtMask  : numSents * tgtMaxLen, indicate where to ignore in the tgtOutput
%  For the monolingual case, each src sent contains a single simple tgtSos,
%   hence srcMaxLen = 1
%
%  Thang Luong @ 2014, 2015, <lmthang@stanford.edu>

  % sent lens
  if length(varargin)==2
    srcLens = varargin{1};
    tgtLens = varargin{2};
  else
    tgtLens = cellfun(@(x) length(x), tgtSents);
    srcLens = cellfun(@(x) length(x), srcSents);
  end
  
  % src
  numSents = length(tgtSents);
  if params.isBi
    srcLens = srcLens + 1; % add eos
    
    % limit sent lengths for training
    if isTest==0
      srcLens(srcLens>params.maxSentLen) = params.maxSentLen; 
    end
    srcMaxLen = max(srcLens);
    srcMinLen = min(srcLens);
  else % mono
    srcMaxLen = 1;
    srcMinLen = 1;
    srcLens = ones(1, numSents);
  end
  
  % tgt
  tgtLens = tgtLens + 1; % add eos
  tgtMaxSentLen = params.maxSentLen;
  if isTest==0 % training
    tgtLens(tgtLens>tgtMaxSentLen) = tgtMaxSentLen; % limit sent lengths
  end
  tgtMaxLen = max(tgtLens);
  
  %% input / output
  if params.isBi
    srcInput = params.srcSos*ones(numSents, srcMaxLen-1);
  end
  tgtInput = [params.tgtSos*ones(numSents, 1) params.tgtEos*ones(numSents, tgtMaxLen-1)];
  tgtOutput = params.tgtEos*ones(numSents, tgtMaxLen);
  
  for ii=1:numSents
    %% IMPORTANT: because we limit sent length, so len(tgtSent) or len(srcSent) 
    %% can be > tgtLen or srcLen, so do not remove 1:tgtLen or 1:srcLen.
    
    % src
    if params.isBi
      srcLen = srcLens(ii)-1; % exclude eos
      srcInput(ii, srcMaxLen-srcLen:srcMaxLen-1) = srcSents{ii}(1:srcLen);      
    end
    
    % tgt
    tgtLen = tgtLens(ii)-1; % exclude eos
    tgtSent = tgtSents{ii};
    
    % tgtEos has been prefilled at the end
    tgtInput(ii, 2:tgtLen+1) = tgtSent(1:tgtLen); % tgtInput: sos word1 ... word_n
    tgtOutput(ii, 1:tgtLen) = tgtSent(1:tgtLen); % tgtOutput: word1 ... word_n eos
  end
  
  % mask
  if params.isBi
    srcMask = srcInput~=params.srcSos;
  end
  tgtMask = tgtInput~=params.tgtEos;
  numWords = sum(tgtMask(:)); 
  
  % assign to data struct
  if params.isBi
    data.srcInput = srcInput;
    data.srcMask = srcMask;
  end
  
  data.srcMaxLen = srcMaxLen;
  data.srcMinLen = srcMinLen;
  data.srcLens = srcLens;
  data.tgtLens = tgtLens;
  data.tgtInput = tgtInput;
  data.tgtOutput = tgtOutput;
  data.tgtMask = tgtMask;
  data.tgtMaxLen = tgtMaxLen;
  data.numWords = numWords;
  
  % sanity check
  if params.assert
    % the last src symbol needs to be eos for all sentences
    if params.isBi
      assert(length(unique(srcInput(:, srcMaxLen)))==1); 
    end
    
    assert(numWords == sum(tgtLens));
  end
end