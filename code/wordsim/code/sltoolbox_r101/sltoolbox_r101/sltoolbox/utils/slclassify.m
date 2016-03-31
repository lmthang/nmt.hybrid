function [decisions, decscores] = slclassify(scores, clabels, op, varargin)
%SLCLASSIFY Classifies a set of samples according to final scores
%
% $ Syntax $
%   - decisions = slclassify(scores, clabels, op, ...)
%   - [decisions, decscores] = slclassify(scores, clabels, op, ...)
%
% $ Arguments $
%   - scores:       the score matrix
%   - clabels:      the class labels of reference samples
%   - op:           the score attribute
%   - decisions:    the classification decisions
%   - decscores:    the scores of the classified targets
%
% $ Description $
%   - decisions = slclassify(scores, clabels, op, ...) classifies
%     a set of query samples to classes. Suppose there are m referenced
%     targets and n query samples, scores should be an m x n matrix, with
%     each column representing the scores of the corresponding sample to
%     all targets.
%     If op is 'high', then the samples are classified to the target of
%     highest score, otherwise, the samples are classified to the target
%     of lowest score.
%     Moreover, you can specify following properties to have more
%     control on the classification process.
%       \*
%       \t   Properties of Classification
%       \h     name     &  description
%             'scheme'  &  The classification scheme
%                          'nn':  using normal nearest sample
%                                 classification (default)
%                          'loo': leave-one-out nearest sample scheme
%                                 (only for the case, when gallery and
%                                  query sets are the same)
%       \*
%
%   - [decisions, decscores] = slclassify(scores, clabels, op, ...)
%     additionally outputs the scores on classified samples.
%
% $ Remarks $
%   - The outputs are 1xn row vectors.
%   - In leave-one-out scheme, it is assumed that the referenced samples
%     and the query samples are actually the same set and in same order.
% 
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%   - Modified by Dahua Lin, on Aug 16th, 2006
%       - eliminate the qlabel parameters, which are essentially not
%         needed.
%       - add functionality to support schemes. In current revision,
%         it supports nearest-neighbor ('nn') and leave-one-out ('loo').
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slclassify', 3);
end
if ~isnumeric(scores) || ndims(scores) ~= 2
    error('sltoolbox:invalidarg', ...
        'scores should be an 2D numeric matrix');
end
m = size(scores, 1);
if length(clabels) ~= m 
    error('sltoolbox:sizmismatch', ...
        'The sizes of labels are inconsistent with the score matrix');
end

opts.scheme = 'nn';
opts = slparseprops(opts, varargin{:});

%% Main skeleton

switch opts.scheme
    case 'nn'
        [decisions, decscores] = classify_nn(scores, clabels, op);
    case 'loo'
        [decisions, decscores] = classify_loo(scores, clabels, op);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid scheme for classification: %s', opts.scheme);
end




%% Decision-making core routines

%% NN

function [decisions, decscores] = classify_nn(scores, clabels, op)

switch op
    case 'high'
        [decscores, decinds] = max(scores, [], 1);
    case 'low'
        [decscores, decinds] = min(scores, [], 1);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid score option %s', op);
end

clabels = clabels(:)';
decisions = clabels(decinds);


%% LOO

function [decisions, decscores] = classify_loo(scores, clabels, op)

n = size(scores, 1);
if size(scores, 2) ~= n
    error('sltoolbox:sizmismatch', ...
        'In leave-one-out scheme, the score matrix should be square');
end
if n < 2
    error('sltoolbox:invalidarg', ...
        'In leave-one-out scheme, the set should have at least two elements');
end

% preprocessing to disable the selection of self
inds_diag = (1 : n+1 : n^2);
switch op
    case 'high'
        scores(inds_diag) = -Inf;
    case 'low'
        scores(inds_diag) = Inf;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid score option %s', op);
end

% classify
switch op
    case 'high'
        [decscores, decinds] = max(scores, [], 1);
    case 'low'
        [decscores, decinds] = min(scores, [], 1);
end

clabels = clabels(:)';
decisions = clabels(decinds);
        
        
        



