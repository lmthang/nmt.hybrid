function [thrs, fars, frrs] = slverifyroc_blks(scores, blocks, labels1, labels2, op, npts)
%SLVERIFYROC_BLKS Computes the verification ROC for blockwise score matrix
%
% $ Syntax $
%   - [thrs, fars, frrs] = slverifyroc_blks(scores, blocks, labels1, labels2, op)
%   - [thrs, fars, frrs] = slverifyroc_blks(scores, blocks, labels1, labels2, op, npts)
% 
% $ Arguments $
%   - scores:       the cell array of array filenames of the scores
%   - blocks:       the division blocks
%   - labels1:      the labels of referred samples
%   - labels2:      the labels of query samples
%   - op:           the option stating the attributes of the scores
%   - npts:         the number of threshold points (default = 500)
%   - thrs:         the sampled thresholds
%   - fars:         the false accept rates at the sampled thresholds
%   - frrs:         the false reject rates at the sampled thresholds
%
% $ Remarks $
%   - This function is an extension of slverifyroc to support blockwise
%     scores in large scale experiments. However, the implementation is
%     fundamentally changed.
%
% $ History $
%   - Created by Dahua Lin, on Aug 8th, 2006
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slverifyroc_blks', 5);
end
if isempty(labels2)
    labels2 = labels1;
end
if ~iscell(scores)
    error('sltoolbox:invalidargs', ...
        'The scores should be a cell array of filenames');
end
if nargin < 6 || isempty(npts)
    npts = 500;
end
nblks = numel(scores);
if ~isequal(size(scores), size(blocks))
    error('The sizes of scores and blocks are inconsistent');
end


%% Collect Histogram

% collect min, max value
maxv = -inf;
minv = inf;
for k = 1 : nblks
    [curmaxv, curminv] = collect_maxmin(slreadarray(scores{k}));
    if curmaxv > maxv
        maxv = curmaxv;
    end
    if curminv < minv
        minv = curminv;
    end
end

if minv >= maxv
    error('sltoolbox:valuerror', ...
        'The minv should be less than maxv');
end

% determine thresholds
thrs = linspace(minv, maxv, npts)';

% collect histograms

H = zeros(npts, 2);
for k = 1 : nblks
    cb = blocks{k};
    l1 = labels1(cb(1,1):cb(2,1));
    l2 = labels2(cb(1,2):cb(2,2));
    curscores = slreadarray(scores{k});    
    curH = collect_scorehist(curscores, l1, l2, thrs);
    
    H = H + curH;
end


%% Compute ROC

hist_a = H(:, 1);
hist_r = H(:, 2);

[thrs, fars, frrs] = slhistroc(hist_a, hist_r, thrs, op);


%% The internal functions

function [maxv, minv] = collect_maxmin(S)

S = S(:);
maxv = max(S);
minv = min(S);


function H = collect_scorehist(S, l1, l2, thrs)
% H is an nbins x 2 matrix stored as [hist_a, hist_r]

m = length(l1);
n = length(l2);
if ~isequal(size(S), [m, n])
    error('sltoolbox:sizmismatch', ...
        'The sizes of labels and score matrix are mismatch');
end

l1 = l1(:);
l2 = l2(:)';
L1 = l1(:, ones(1, n));
L2 = l2(ones(m, 1), :);
signals = (L1 == L2);
clear L1 L2;

scores_a = S(signals);
scores_r = S(~signals);

if ~isempty(scores_a)
    hist_a = histc(scores_a, thrs);
else
    hist_a = zeros(length(thrs), 1);
end

if ~isempty(scores_r)
    hist_r = histc(scores_r, thrs);
else
    hist_r = zeros(length(thrs), 1);
end

H = [hist_a, hist_r];


