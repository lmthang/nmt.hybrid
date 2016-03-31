function [thrs, fars, frrs] = slhistroc(hist_a, hist_r, sepvals, op)
%SLHISTROC Computes the ROC curve from value histogram
%
% $ Syntax $
%   - [thrs, fas, frs] = slhistroc(hist_a, hist_r, sepvals, op)
%
% $ Arguments $
%   - hist_a:       the histogram of the values that should be accepted
%   - hist_r:       the histogram of the values that should be rejected
%   - sepvals:      the separation values of the histograms
%   - op:           the option of the attributes of the values
%   - thrs:         the sampled threshold values
%   - fars:         the false accept rates at the sampled thresholds
%   - frrs:         the false reject rates at the sampled thresholds
%
% $ Description $
%   - [thrs, fas, frs] = slhistroc(hist_a, hist_r, sepvals, op) computes
%     the ROC curve based on histograms. The hist_a and hist_r should have
%     corresponding bins, and the bins are sorted in ascending order of
%     the values they represent. Suppose the number of bins be n, then 
%     sepvals are the values on the bin edges, and the length of sepvals
%     should be n. Note that the histtograms should be created following
%     the rules as in histc, so the last element of histogram is the 
%     number of elements that exactly match the last sep value.
%     The op can be either 'low' or 'high', when it is 'low', the value
%     lower than threshold would be accepted, otherwise the value higher
%     than threshold would be accepted.
%     The output thrs is just sepvals, while fars and frrs are 
%     corresponding false accept rates and false reject rates, with 
%     n = nbins + 1 elements.
%
% $ History $
%   - Created by Dahua Lin, on Aug 8th, 2006
%

%% parse and verify arguments

if nargin < 4
    raise_lackinput('slhistroc', 4);
end

nbins = length(hist_a);
if length(hist_r) ~= nbins
    error('sltoolbox:sizmismatch', ...
        'The sizes of hist_a and hist_r are inconsistent');
end
if length(sepvals) ~= nbins
    error('sltoolbox:sizmismatch', ...
        'The length of sepvals should be number of bins');
end


%% Compute ROC

% preprocess
hist_r(end-1) = hist_r(end-1) + hist_r(end);
hist_a(end-1) = hist_a(end-1) + hist_a(end);
hist_r = hist_r(1:end-1);
hist_a = hist_a(1:end-1);

hist_r = hist_r(:);
hist_a = hist_a(:);
thrs = sepvals(:);

nr = sum(hist_r);
na = sum(hist_a);


switch op
    case 'low'
        fars = [0; cumsum(hist_r)] / nr;
        frrs = [na; na - cumsum(hist_a)] / na;
        
    case 'high'
        fars = [nr; nr - cumsum(hist_r)] / nr;
        frrs = [0; cumsum(hist_a)] / na;
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid option %s for roc', op);        
end
    



