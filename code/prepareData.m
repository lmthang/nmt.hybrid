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
    if isTest==0 || params.attnFunc==1 % limit sent lengths for training or for attention model during both training/testing
      srcLens(srcLens>params.maxSentLen) = params.maxSentLen; 
    end
    srcMaxLen = max(srcLens);
  else
    srcMaxLen = 1;
    srcLens = ones(1, numSents);
  end
  
  % tgt
  tgtLens = tgtLens + 1; % add eos
  tgtMaxSentLen = params.maxSentLen;
  if params.posModel>0
    tgtMaxSentLen = (tgtMaxSentLen-1)*2 + 1;
  end
  if isTest==0 % training
    tgtLens(tgtLens>tgtMaxSentLen) = tgtMaxSentLen; % limit sent lengths
    tgtMaxLen = max(tgtLens);
  else
    tgtMaxLen = max(tgtLens);
  end
  
  %% input / output
  if params.isBi
    srcInput = params.srcZero*ones(numSents, srcMaxLen);
  end
  tgtInput = [params.tgtSos*ones(numSents, 1) params.tgtEos*ones(numSents, tgtMaxLen-1)];
  tgtOutput = params.tgtEos*ones(numSents, tgtMaxLen);
  
  for ii=1:numSents
    %% src
    if params.isBi
      srcLen = srcLens(ii);
      
      if params.separateEmb==1 % separate vocab
        srcInput(ii, srcMaxLen-srcLen+1:srcMaxLen-1) = srcSents{ii}(1:srcLen-1);
      else
        srcInput(ii, srcMaxLen-srcLen+1:srcMaxLen-1) = srcSents{ii}(1:srcLen-1) + params.tgtVocabSize;
      end
      
      srcInput(ii, srcMaxLen) = params.srcEos;
    end
    
    %% tgt
    tgtSent = tgtSents{ii};
    tgtLen = tgtLens(ii);
    
    % tgtEos has been prefilled
    tgtInput(ii, 2:tgtLen) = tgtSent(1:tgtLen-1);
    tgtOutput(ii, 1:tgtLen-1) = tgtSent(1:tgtLen-1);
  end
  
  % mask
  if params.isBi
    srcMask = srcInput~=params.srcZero;
  end
  tgtMask = tgtInput~=params.tgtEos;
  numWords = sum(tgtMask(:)); 
  
  % positional models
  if params.posModel>0
    % tgt sent: pos1 word1 ... pos_n word_n <eos>. 
    % posOutput: pos1 ... pos_n <eos>
    data.posOutput = tgtOutput(:, 1:2:tgtMaxLen);
    data.posMask = tgtMask(:, 1:2:tgtMaxLen);
  end
  
  % assign to data struct
  if params.isBi
    data.srcInput = srcInput;
    data.srcMask = srcMask;
    
%     data.input = [srcInput tgtInput(:, 2:end)];
%     data.inputMask = [srcMask tgtMask(:, 2:end)];
    data.input = [srcInput(:, 1:end-1) tgtInput]; % tgtInput starts with tgtSos so as to be compatible with mono language models
    data.inputMask = [srcMask(:, 1:end-1) tgtMask];
  else
    data.input = tgtInput;
    data.inputMask = tgtMask;
  end
  
  data.srcMaxLen = srcMaxLen;
  data.srcLens = srcLens;
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
      assert(srcInput(1, srcMaxLen)==params.srcEos);
    end
    
    assert(numWords == sum(tgtLens));
  end
end


    %label = 'input';
    %printSent(input(1, :), params.vocab, ['  ', label, ' 1:']);
    %printSent(input(end, :), params.vocab, ['  ', label, ' end:']);
    
