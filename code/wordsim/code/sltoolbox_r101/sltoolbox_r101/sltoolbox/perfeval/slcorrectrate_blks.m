function cr = slcorrectrate_blks(scores, blocks, clabels, qlabels, op, varargin)
%SLCORRECTRATE_BLKS Computes the correct rate based on blockwise scores
%
% $ Syntax $
%   - cr = slcorrectrate_blks(scores, blocks, clabels, qlabels, op, ...)
%
% $ Arguments $
%   - scores:       the score matrix
%   - blocks:       the cell array of block limits
%   - clabels:      the class labels of reference samples
%   - qlabels:      the query labels of query samples
%   - op:           the score attribute  
%   - cr:           the computed classification correct rate
%
% $ Remarks $
%   - An extension of slcorrectrate to support blockwise scores
%
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%   - Modified by Dahua Lin on Aug 16th, 2006
%     - Based on new slclassify to support multiple schemes
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slcorrectrate_blks', 5);
end


%% Make decision

n = length(qlabels);
decisions = slclassify_blks(scores, n, blocks, clabels, op, varargin{:});

%% Evaluate correct rate

qlabels = qlabels(:)';
cr = sum(decisions == qlabels) / length(qlabels);



