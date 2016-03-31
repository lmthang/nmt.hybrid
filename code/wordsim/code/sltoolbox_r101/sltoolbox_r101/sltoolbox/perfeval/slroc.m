function [thrs, fars, frrs] = slroc(scores, signs, thres, op)
%SLROC Computes the ROC
%
% $ Syntax $
%   - [thrs, fars, frrs] = slroc(scores, signs, thres, op) 
%
% $ Arguments $
%   - scores:           the scores representing the signal intensities
%   - signs:            the signs representing the groundtruth (0 or 1)
%   - thres:            the descriptor indicating how to sample the 
%                       thresholds at which the rate is computed
%   - op:               the option stating the attributes of the scores.
%   - thrs:             the sampled threshold values
%   - fars:             the false accept rates at the sampled thresholds
%   - frrs:             the false reject rates at the sampled thresholds
% 
% $ Description $
%   - [thrs, fars, frrs] = slroc(scores, signs, thres, op) Computes the 
%     ROC of a receiver from the scores and groudtruth signs. The argument 
%     thres specifies the sampled thresholds at which the false accept 
%     rate and false reject rate is evaluated. If thres is an integer, 
%     say n, then n equal-interval integers from lowest to highest scores 
%     are taken as samples. The thres can also be a vector containing the 
%     sampled thresholds, which should be arranged in ascending order.
%     op states the attributes of the scores, which
%     takes either of the two values: 'high' or 'low'. If op is 'high', a 
%     higher score indicates a better match; if op is 'low', a lower score
%     indicates a better match. The cumulative score will be computed
%     up to the number of all classes.
%     For output, if n thresholds are sampled, then thrs, fars, and frrs
%     for all n x 1 vectors, containing the sampled threshold values,
%     and the corresponding false accept rates, and false reject rates.
%
% $ History $
%   - Created by Dahua Lin on Jun 9th, 2005
%   - Modified by Dahua Lin on May 1st, 2006
%     - Base on the sltoolbox v4
%   - Modified by Dahua Lin on Aug 8th, 2006
%     - Base on slhistroc
%   

%% parse and verify the input arguments
if nargin < 4
    raise_lackinput('slroc', 4);
end
if ~isequal(size(scores), size(signs))
    error('sltoolbox:sizmismatch', ...
        'The sizes of scores and signs are not match');
end

% the following two statements are disabled in the 2006-08-08 modification
% since scores(signs) will automatic serialize the values
% there is no need to write this statement, just a waste of time and mem
% scores = scores(:); 
% signs = logical(signs(:));

if numel(thres) == 1
    n = thres;
    highscore = max(max(scores));
    lowscore = min(min(scores));
    thrs = linspace(lowscore, highscore, n)';
else
    thrs = thres(:);
end

    

%% compute
scores_a = scores(signs);
scores_r = scores(~signs);

hist_a = histc(scores_a, thrs);
hist_r = histc(scores_r, thrs);

[thrs, fars, frrs] = slhistroc(hist_a, hist_r, thrs, op);





