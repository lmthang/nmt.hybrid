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
  
  numSents = length(tgtSents);
  if params.isBi
    srcLens = srcLens + 1; % add eos
    if isTest==0 || params.attnFunc==1 % limit sent lengths for training or for attention model during both training/testing
      srcLens(srcLens>params.maxSentLen) = params.maxSentLen; 
      srcMaxLen = params.maxSentLen;
    else
      srcMaxLen = max(srcLens);
    end
  end
  
  tgtLens = tgtLens + 1; % add eos

  if isTest==0 % training
    tgtLens(tgtLens>params.maxSentLen) = params.maxSentLen; % limit sent lengths
    tgtMaxLen = max(tgtLens);
    assert(tgtMaxLen<=params.maxSentLen);
  else
    tgtMaxLen = max(tgtLens);
  end

  % positional models
  if params.posModel>0
    % tgt sent: pos1 word1 ... pos_n word_n <eos>. 
    % srcPos: pos1 ... pos_n
    srcPos = params.eosPosId*ones(numSents, floor(tgtMaxLen/2)); 
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
      srcInput(ii, srcMaxLen-srcLen+1:srcMaxLen-1) = srcSents{ii}(1:srcLen-1) + params.tgtVocabSize;
      srcInput(ii, srcMaxLen) = params.srcEos;
    end
    
    %% tgt
    tgtSent = tgtSents{ii};
    tgtLen = tgtLens(ii);
    
    % tgtEos has been prefilled
    tgtInput(ii, 2:tgtLen) = tgtSent(1:tgtLen-1);
    tgtOutput(ii, 1:tgtLen-1) = tgtSent(1:tgtLen-1);
    
    % get src positions
    if params.posModel>0
      srcPos(ii, 1:tgtLen-1) = tgtSent(1:2:tgtLen-1);
    end
  end
  
  if params.isBi
    srcMask = srcInput~=params.srcZero;
  end
  tgtMask = tgtInput~=params.tgtEos;
  numWords = sum(tgtMask(:)); 
  
  % sanity check
  if params.assert
    % the last src symbol needs to be eos for all sentences
    if params.isBi
      assert(length(unique(srcInput(:, srcMaxLen)))==1); 
      assert(srcInput(1, srcMaxLen)==params.srcEos);
    end
    
    assert(numWords == sum(tgtLens));
  end
  
  if params.isBi
    data.srcInput = srcInput;
    data.srcMask = srcMask;
    
    data.input = [srcInput tgtInput(:, 2:end)];
    data.inputMask = [srcMask tgtMask(:, 2:end)];
  else
    data.input = trainData.tgtInput;
    data.inputMask = trainData.tgtMask;
  end
  
  data.tgtInput = tgtInput;
  data.tgtOutput = tgtOutput;
  data.tgtMask = tgtMask;
  
  data.srcMaxLen = srcMaxLen;
  data.tgtMaxLen = tgtMaxLen;
  data.numWords = numWords;
  data.srcLens = srcLens;
end

    
%     if params.posModel>0 % add an extra <s_eos> to the src side
%       srcMaxLen = srcMaxLen+1;
%     end

%   % positional models, tgt sent: pos1 word1 ... pos_n word_n <eos>
%   if params.posModel>0
%     tgtLens = (tgtLens+1)/2;
%   end

%       if params.posModel>0 % add an extra <s_eos> to the src side
%         input(ii, srcMaxLen-srcLen:srcMaxLen-1) = srcSents{ii}(1:srcLen) + params.tgtVocabSize; % src part
%         input(ii, srcMaxLen) = params.srcEosVocabId;
%       else
%         input(ii, srcMaxLen-srcLen+1:srcMaxLen) = srcSents{ii}(1:srcLen) + params.tgtVocabSize; % src part
%       end


%   % positional models
%   if params.posModel>0
%     data.srcPos = srcPos;
%   end


%       if params.inputFormat==1 % left-aligned
%         srcInput(ii, 1:srcLen-1) = srcSents{ii}(1:srcLen-1) + params.tgtVocabSize;
%         srcInput(ii, srcLen) = params.srcEos;
%       else
%         
%       end

    %label = 'input';
    %printSent(input(1, :), params.vocab, ['  ', label, ' 1:']);
    %printSent(input(end, :), params.vocab, ['  ', label, ' end:']);
    
