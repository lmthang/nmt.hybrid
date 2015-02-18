function [sents, numSents, sentLens] = loadBatchData(fid, baseIndex, batchSize, varargin)
%%%
%
% Load a number of sentences (integers per line) from a file.
%   baseIndex: minimum integer value in the input file.
%   batchSize: number of sents to read (if batchSize==-1, read all).
%
% Thang Luong @ 2013, <lmthang@stanford.edu>
%%%

  if length(varargin)==1
    suffix = varargin{1};
  else
    suffix = [];
  end
  
  sents = cell(1, batchSize);
  sentLens = zeros(1, batchSize);
  numSents = 0;
  while ~feof(fid)
    indices = sscanf(fgetl(fid), '%d') + (1-baseIndex);
    if isempty(indices) % ignore empty lines
      continue
    end
    
    numSents = numSents + 1;
    sents{numSents} = [indices' suffix];
    sentLens(numSents) = length(sents{numSents});
    if numSents==batchSize
      break;
    end
  end
  
  % delete empty values
  sents((numSents+1):end) = []; 
  sentLens((numSents+1):end) = [];
end
