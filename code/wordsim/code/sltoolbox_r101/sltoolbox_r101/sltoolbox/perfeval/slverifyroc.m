function [thrs, fars, frrs] = slverifyroc(scores, labels1, labels2, op, npts)
%SLVERIFYROC Computes the verification ROC
%
% $ Syntax $
%   - [thrs, fars, frrs] = slverifyroc(scores, labels1, labels2, op)
%   - [thrs, fars, frrs] = slverifyroc(scores, labels1, labels2, op, npts)
% 
% $ Arguments $
%   - scores:       the score matrix for verification
%   - labels1:      the labels of referred samples
%   - labels2:      the labels of query samples
%   - op:           the option stating the attributes of the scores
%   - npts:         the number of threshold points (default = 200)
%   - thrs:         the sampled thresholds
%   - fars:         the false accept rates at the sampled thresholds
%   - frrs:         the false reject rates at the sampled thresholds
%
% $ Description $
%   - [thrs, fars, frrs] = slverifyroc(scores, labels1, labels2, op) 
%     Computes the verification ROC based on the pairwise scores. 
%     If there are m samples in the first set, n samples in the second set, 
%     then the scores matrix should be of size m x n, and the labels1 and 
%     labels2 should be a vector of m and n elements respectively. 
%     If labels2 is given empty, then it is supposed to be the same as 
%     labels1, where we are performing self-pairwise-verification.
%     Here op can be 'low' or 'high', if op is 'high' means that a higher
%     score indicating higher similarity, vice versa for 'low' op. 
%     For output, if n thresholds are sampled, then thrs, fars, and frrs
%     for all n x 1 vectors, containing the sampled threshold values,
%     and the corresponding false accept rates, and false reject rates.
%
%   - [thrs, fars, frrs] = slverifyroc(scores, labels1, labels2, op, npts)
%     You can specify the number of threshold points to sampled by npts.
%
% $ History $
%   - Created by Dahua Lin on Jun 10th, 2005
%   - Modified by Dahua Lin on May 1st, 2006
%     - Base on the sltoolbox v4
%   - Modified by Dahua Lin on Aug 8th, 2006
%     - Add one more argument npts to tune the density of sampling
% 

%% parse and verify input arguments
if nargin < 4
    raise_lackinput('slverifyroc', 4);
end
if isempty(labels2)
    labels2 = labels1;
end
if ndims(scores) ~= 2
    error('sltoolbox:invaliddims', ...
        'The scores should be a 2D matrix');
end
[m, n] = size(scores);
if length(labels1) ~= m || length(labels2) ~= n 
    error('The sizes of labels do not match that of scores');
end

if nargin < 5 || isempty(npts)
    npts = 200;
end


%% generate the signs
labels1 = labels1(:);
labels2 = labels2(:)';
L1 = labels1(:, ones(1, n));
L2 = labels2(ones(m, 1), :);
signals = (L1 == L2);
clear L1 L2;

%% compute
[thrs, fars, frrs] = slroc(scores, signals, npts, op);



    
