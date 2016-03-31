function cs = slcumuscore(scores, clabels, qlabels, op, maxr)
%SLCUMUSCORE Computes the cumulative score on multi-class classification
%
% $ Syntax $
%   - cs = slcumuscore(scores, clabels, qlabels, op)
%   - cs = slcumuscore(scores, clabels, qlabels, op, maxr)
%
% $ Arguments $
%   - scores:           the scores to support the classification
%   - clabels:          the labels of classes
%   - qlabels:          the groundtruth of the labels of query samples
%   - op:               the option of the score
%   - maxr:             the maximum number of ranked classes
%   - cs:               the matrix of cumulative scores
%
% $ Description $
%   - cs = slcumuscore(scores, clabels, qlabels, op) Computes the 
%     cumulative scores of score-based classification. Suppose there 
%     are m classes and n query samples to be classified. Then scores
%     should be an m x n matrix with the entry at the i-th row and j-th
%     column representing the score of the j-th sample belonging to the
%     the i-th class. clabels and qlabels should be length-m and length-n
%     vectors respectively. op states the attributes of the scores, which
%     takes either of the two values: 'high' or 'low'. If op is 'high', a 
%     higher score indicates a better match; if op is 'low', a lower score
%     indicates a better match. The cumulative score will be computed
%     up to the number of all classes.
%
%   - cs = slcumuscore(scores, clabels, qlabels, op, maxr) computes the
%     cumulative scores of score-based classification up to the number 
%     of classes specified by maxr.
%
% $ History $
%   - Created by Dahua Lin on Jun 10th, 2005
%   - Modified by Dahua Lin on May 1st, 2006
%     - Base on the sltoolbox v4
%

%% parse and verify input arguments
if nargin < 4
    raise_lackinput('slcumuscore', 4);
end
if ndims(scores) ~= 2
    error('sltoolbox:invaliddims', ...
        'The matrix scores should be a 2D matrix');
end
[nclasses, nsamples] = size(scores);
qlabels = qlabels(:);
clabels = clabels(:);
if length(clabels) ~= nclasses || length(qlabels) ~= nsamples
    error('sltoolbox:sizmismatch', ...
        'the labels vectors do not match the size of scores');
end

if nargin < 5 || isempty(maxr)
    maxr = nclasses;
end

%% compute
switch op
    case 'low'
        [ss, sp] = sort(scores, 1, 'ascend');
    case 'high'
        [ss, sp] = sort(scores, 1, 'descend');
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid option %s for slcumuscore', op);
end
slignorevars(ss);

if maxr < nclasses
    sp = sp(1:maxr, :);
else
    maxr = nclasses;
end
decisions = clabels(sp);
matches = (decisions == repmat(qlabels', [maxr, 1]));

cs = cumsum(sum(matches, 2), 1) / nsamples;


