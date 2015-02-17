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
function [data] = prepareData(srcSents, tgtSents, params, varargin)
  if length(varargin)==2
    srcLens = varargin{1};
    tgtLens = varargin{2};
  else
    srcLens = cellfun(@(x) length(x), srcSents);
    tgtLens = cellfun(@(x) length(x), tgtSents);
%     if params.isBi
%       if size(srcLens, 2)==1 % we want row vectors
%         srcLens = srcLens';
%       end
%     end
  end
  
  numSents = length(tgtSents);
  if params.isBi
    srcZeroId = params.tgtVocabSize + params.srcSos;
    srcMaxLen = max(srcLens);
    
    % attention model
    if params.attnFunc>0 && srcMaxLen > params.maxSentLen 
      fprintf(2, 'prepareData: change srcMaxLen from %d -> %d\n', srcMaxLen, params.maxSentLen);
      srcMaxLen = params.maxSentLen;
    end
  else
    srcLens = ones(numSents, 1);
    srcZeroId = params.tgtSos;
    srcMaxLen = 1;
  end
  tgtMaxLen = max(tgtLens);
  input = [srcZeroId*ones(numSents, srcMaxLen) params.tgtEos*ones(numSents, tgtMaxLen-1)]; % size numSents * (srcMaxLen + tgtMaxLen - 1)
  tgtOutput = params.tgtEos*ones(numSents, tgtMaxLen);
  
  % positional models
  if params.posModel==2 || params.posModel==3 
    srcPos = params.tgtEos*ones(numSents, (tgtMaxLen+1)/2); % since tgt sent: pos1 word1 ... pos_n word_n <eos>. Later we want: pos1 ... pos_n pos_eos.
  end
  
  for ii=1:numSents
    %% src
    if params.isBi
      srcLen = srcLens(ii);
      if params.attnFunc>0 && srcLen>srcMaxLen % attention model
        srcLen = srcMaxLen;
      end
      input(ii, srcMaxLen-srcLen+1:srcMaxLen) = srcSents{ii}(1:srcLen) + params.tgtVocabSize; % src part
    end
    
    %% tgt
    tgtSent = tgtSents{ii};
    
    % positional models
    if params.posModel==2 || params.posModel==3 
      % words
      tgtSent(1:2:end-1) = []; % remove positions
      tgtLens(ii) = length(tgtSent);
      
      % positions
      positions = tgtSents{ii}(1:2:end-1);
      srcPos(ii, 1:tgtLens(ii)-1) = (1:tgtLens(ii)-1) - (positions-params.zeroPosId); % src_pos = tgt_pos - relative_pos
      srcPos(ii, tgtLens(ii)) = srcLen; % <eos>
    else
      
    end
    
    input(ii, srcMaxLen+1:srcMaxLen+tgtLens(ii)-1) = tgtSent(1:end-1); % tgt part
    tgtOutput(ii, 1:tgtLens(ii)) = tgtSent;
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
    %label = 'input';
    %printSent(input(1, :), params.vocab, ['  ', label, ' 1:']);
    %printSent(input(end, :), params.vocab, ['  ', label, ' end:']);
  end
  
  data.input = input;
  data.inputMask = inputMask;
  data.tgtOutput = tgtOutput;
  data.srcMaxLen = srcMaxLen;
  data.tgtMaxLen = tgtMaxLen;
  data.numWords = numWords;
  data.srcLens = srcLens;
  
  % positional models
  if params.posModel==2 || params.posModel==3 
    srcPos(srcPos<0) = 0;
    data.srcPos = srcPos;
  end
end
