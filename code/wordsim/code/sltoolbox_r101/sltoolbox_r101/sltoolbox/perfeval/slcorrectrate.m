function cr = slcorrectrate(scores, clabels, qlabels, op, varargin)
%SLCORRECTRATE Computes the correct rate of classification
%
% $ Syntax $
%   - cr = slcorrectrate(scores, clabels, qlabels, op, ...) 
%
% $ Arguments $
%   - scores:           the scores to support the classification
%   - clabels:          the labels of classes
%   - qlabels:          the groundtruth of the labels of query samples
%   - op:               the option of the score
%   - cr:               the correct rate of the score-based classification
%
% $ Description $
%   - cr = slcorrectrate(scores, clabels, slabels, op, ...) 
%     Computes the classification correct rate. Suppose we want to 
%     classify n samples into m classes, then scores will be an m x n 
%     matrix, with the entry at i-th row, j-th column representing the 
%     score of the j-th sample in the i-th class. 
%     For op, it can take either of the two values: 'high' and 'low'. 
%     If op is 'high' then it means that the sample will be classified 
%     to the class where it has the highest score value, vice versa 
%     for 'low'.
% 
% $ History $
%   - Created by Dahua Lin on Jun 10th, 2005
%   - Modified by Dahua Lin on May 1st, 2005
%     - To base on the sltoolbox v4.
%   - Modified by Dahua Lin on Aug 9th, 2006
%     - Extract slclassify as independent function and base on it
%   - Modified by Dahua Lin on Aug 16th, 2006
%     - Based on new slclassify to support multiple schemes
%

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('slcorrectrate', 4);
end


%% Make decision

decisions = slclassify(scores, clabels, op, varargin{:});

%% Evaluate correct rate

qlabels = qlabels(:)';
cr = sum(decisions == qlabels) / length(qlabels);







