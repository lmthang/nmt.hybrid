function [batch, mask, maxLen, numSeqs] = leftPad(seqs, padSymbol, varargin)
% Left padding the data so that all sequences have the same length.
% Optionally, we can append the eos symbol through the third argument.
% Input:
%   seqs: cell array, sequences of different lengths.
%   padSymbol: to left pad.
%   varargin: optional to specify the eos symbol
% Output:
%   batch: left-padded data structure.
%   mask: 0-1 matrix, same size as batch, 0 at those padding positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

  numSeqs = length(seqs);
  lens = cellfun(@(x) length(x), seqs);
  maxLen = max(lens);
  
  % append eos
  eos = 0;
  if ~isempty(varargin)
    eos = varargin{1};
    maxLen = maxLen + 1;
  end
  
  batch = padSymbol*ones(numSeqs, maxLen);
  for ii=1:numSeqs
    len = lens(ii);
    
    if eos > 0
      batch(ii, end-len:end-1) = seqs{ii}(1:len);
      batch(ii, end) = eos;
    else
      batch(ii, end-len+1:end) = seqs{ii}(1:len);      
    end
  end
  mask = batch ~= padSymbol;
end