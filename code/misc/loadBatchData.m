function [sents, numSents] = loadBatchData(fid, baseIndex, batchSize, suffix)
%%%
%
% Load a number of sentences (integers per line) from a file.
%   baseIndex: minimum integer value in the input file.
%   batchSize: number of sents to read (if batchSize==-1, read all).
%
% Thang Luong @ 2013, <lmthang@stanford.edu>
%%%
  
  if ~exist('suffix', 'var')
    suffix = [];
  end
  
  sents = cell(batchSize, 1);
  numSents = 0;
  while ~feof(fid)
    indices = sscanf(fgetl(fid), '%d') + (1-baseIndex);
    if isempty(indices) % ignore empty lines
      continue
    end
    
    numSents = numSents + 1;
    sents{numSents} = [indices' suffix];
    if numSents==batchSize
      break;
    end
  end
  sents((numSents+1):end) = []; % delete empty cells
end
