function [data] = prepareMonoData(sents, sos, eos, maxLen, isSrc) %, varargin)
%  Organize data into matrix format and produce masks.
%
%  Thang Luong @ 2016, <lmthang@stanford.edu>

%   % sent lens
%   if length(varargin)==1
%     lens = varargin{1};
%   else
%     lens = cellfun(@(x) length(x), sents);
%   end
  
  lens = cellfun(@(x) length(x), sents);
  
  if isSrc % left pad
    [data.srcInput, data.srcMask, data.srcMaxLen, data.numSents] = leftPad(sents, lens, sos, eos, maxLen);
  else % right pad
    [data.tgtInput, data.tgtMask, data.tgtMaxLen, data.numSents] = rightPad(sents, lens, eos, sos, maxLen);
    data.tgtOutput = [data.tgtInput(:, 2:end) eos*ones(data.numSents, 1)];
  end
end