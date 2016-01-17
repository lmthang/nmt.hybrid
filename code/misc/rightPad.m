function [batch, mask, maxLen, numSeqs] = rightPad(seqs, lens, padSymbol, varargin)
% Right padding the data so that all sequences have the same length.
% Optionally, we can append the sos symbol through the third argument.
% Input:
%   seqs: cell array, sequences of different lengths.
%   padSymbol: to left pad.
%   varargin: optional to specify the sos symbol
% Output:
%   batch: right-padded data structure.
%   mask: 0-1 matrix, same size as batch, 0 at those padding positions.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>

  numSeqs = length(seqs);
  % lens = cellfun(@(x) length(x), seqs);
  maxLen = max(lens);
  
  % append sos
  sos = 0;
  if ~isempty(varargin)
    sos = varargin{1};
    maxLen = maxLen + 1;
  end
  
  batch = padSymbol*ones(numSeqs, maxLen);
  for ii=1:numSeqs
    len = lens(ii);
    
    if sos > 0
      batch(ii, 1) = sos;
      batch(ii, 2:len+1) = seqs{ii}(1:len);
    else
      batch(ii, 1:len) = seqs{ii}(1:len);      
    end
  end
  mask = batch ~= padSymbol;
end