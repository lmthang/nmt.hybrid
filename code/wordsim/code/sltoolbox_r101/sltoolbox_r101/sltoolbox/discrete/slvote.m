function H = slvote(models, m, samples, n, evalfunctor, countrule, varargin)
%SLVOTE Builds histogram by voting (or fuzzy voting)
%
% $ Syntax $
%   - H = slvote(models, m, samples, n, evalfunctor, countrule, ...)
%
% $ Arguments $
%   - models:       The models to be voted for
%   - m:            The number of models
%   - samples:      The samples (as voters)
%   - n:            The number of samples
%   - evalfunctor:  The functor to evaluate the votes for samples
%                   it should be like the form:
%                       V = f(models, samples, ...)
%                   The form of V depends on count rule.
%   - countrule:    The rule of counting the votes.
%   - H:            The built histogram of votes on the models
%
% $ Description $
%   - H = slvote(models, m, samples, n, evalfunctor, countrule, ...) makes 
%     histogram on the models using the specified voting method. The basic
%     procedure consists of two stages. The first stage is to use the
%     evalfunctor to evaluate the votes, and then the histogram is built
%     using the votes according to the specified counting rule. 
%     If there are m models, then H would be an m x 1 column vector.
%     
%     This function supports a series of counting ways for voting. 
%     Correspondingly, the evalfunctor should have different format of 
%     output for different rules.

%     You can further specify the following properties:
%       - 'weights':        The weights of samples. They will be multiplied
%                           to the contributions of the samples. 
%                           (default = [], if specified, it is 1 x n row)
%       - 'normalized':     Whether to normalize the histogram so that the
%                           sum of the votings to all bins are normalized
%                           to 1. (default = false)
%  
% $ History $
%   - Created by Dahua Lin, on Sep 17, 2006
%   

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('slvote', 6);
end

opts.weights = [];
opts.normalized = false;
opts = slparseprops(opts, varargin{:});

%% main skeleton

% make vote
V = slevalfunctor(evalfunctor, models, samples);

% make histogram
H = slcountvote(m, n, V, opts.weights, countrule);

% normalize the histogram
if opts.normalized
    H = H / sum(H);
end

