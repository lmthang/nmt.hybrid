function H = slvechist(X0, X, varargin)
%SLVECHIST Makes the histogram on prototype vectors by voting
%
% $ Syntax $
%   - H = slvechist(X0, X, ...)
%
% $ Arguments $
%   - X0:       The sample matrix of the prototypes (be voted)
%   - X:        The samples to vote (voters)
%   - H:        The resultant histogram 
%
% $ Description $
%   - H = slvechist(X0, X, ...) makes the histogram on vectors by counting
%     how many of X belong to each of X0, or evaluating the confidences
%     of the ownership. If there are m samples in X0, then H would be a
%     m x 1 column vector. The process consists of two stages: first, it
%     evaluates which sample belongs to which prototype and also the 
%     confidences if necessary, this stage is called voting, then the 
%     histogram would be built based on the voting results.
%     You can specify the following properties to control the process:
%       - 'countrule':      The rule of counting the votes (default = 'nm')
%                           Please refer to slcountvote for a list of
%                           available counting rules.
%       - 'clsmethod':      The method of classifying the vectors to 
%                           the prototypes. (default = 'pwcomp')
%                           - 'pwcomp': computing the metrics between 
%                             all samples and all prototypes pairwisely.
%                             then select the most close ones
%                           - 'kdtree': using KD-tree to select the most
%                             close ones.
%                           This property only takes effect when countrule
%                           is 'nm' or 'nmx'.
%       - 'nnparams'        The parameters to find the neighboring 
%                           prototypes in the form {method, ...}. Please
%                           refer to slfindnn for details.
%                           This property only takes effect when countrule
%                           is 'mmc', 'mms' or 'mmns'.
%                           (default = [], means to use all prototypes as
%                            contributive neighbors)
%       - 'metric'          The type of distances to compare vectors.
%                           It can be a string representing the name of
%                           the metric type, or a cell array of
%                           parameters given as {method, ...} for 
%                           parameterized metric computation. Please refer
%                           to slmetric_pw for a list of available methods
%                           and the specification of the parameters.
%                           You can also define your own distance by using 
%                           a distance computing function handle here, it
%                           is invoked using the syntax:
%                               D = f(X1, X2) 
%                           to compute pairwise distances. 
%                           (default = 'eucdist')
%       - 'cfunc'           The functor to evaluate likelihood (confidence)
%                           values based on metric values.
%                           This property only takes effect when countrule
%                           uses confidences values. 
%                               C = f(V)
%                           when countrule is 'nmx', the input V is a 1 x n                          
%                           row vector, otherwise the input V is a m x n 
%                           matrix (or sparse matrix). The output should
%                           be of the same size as V, and translate the
%                           metric values to the confidence values in the
%                           corresponding positions. 
%                           Please note that the input to f contains all
%                           metric values, actually you can use their 
%                           integral attributes and relationship in the
%                           computation of confidence values.
%                           (default = [], for the countrule using 
%                           confidences, it must be specified. If you 
%                           would like to just use the metric values as
%                           confidences just set cfunc to 'um')
%       - 'evalfunctor'     The user-supplied functor to do voting. If
%                           it is specified the function just invokes the
%                           functor to do voting and ignore other options.
%                           It would be invoked as
%                               V = f(X0, X, ...)
%                           (default = [])
%       - 'weights':        The weights of samples. They will be multiplied
%                           to the contributions of the samples. 
%                           (default = [], if specified, it is 1 x n row)
%       - 'normalized':     Whether to normalize the histogram so that the
%                           sum of the votings to all bins are normalized
%                           to 1. (default = false)
% 
% $ Remarks $
%   - The metric specification will not take effect for KD-tree (ANN) based
%     methods, they can only use Euclidean distances.
%
%   - When counting rule uses one-best-prototype strategy, such as 'nm'
%     and 'nmx', the function uses clsmethod to classify samples to 
%     the best prototypes, otherwise, it uses multi-best-prototype strategy
%     the function uses nnparams to construct the neighborhood graph.
%     When nnparams is [], all prototypes will be considered as
%     neighborhood of all samples, then all pairwise relationship will be
%     utilized.
%
% $ History $
%   - Created by Dahua Lin on Sep 18th, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slvechist', 2);
end

if ~isnumeric(X0) || ndims(X0) ~= 2
    error('sltoolbox:invalidarg', 'X0 should be a 2D numeric matrix');
end
if ~isnumeric(X) || ndims(X) ~= 2
    error('sltoolbox:invalidarg', 'X should be a 2D numeric matrix');
end

[d, m] = size(X0);
[d2, n] = size(X);
if d ~= d2
    error('sltoolbox:sizmismatch', 'The dimensions of X and X0 are mismatched');
end

    
opts.countrule = 'nm';
opts.clsmethod = 'pwcomp';
opts.nnparams = [];
opts.metric = 'eucdist';
opts.cfunc = [];
opts.evalfunctor = [];
opts.weights = [];
opts.normalized = false;
opts = slparseprops(opts, varargin{:});


%% main skeleton

% decide scheme
% Vform:
%   - 0:    1 x n or 2 x n matrix
%   - 1:    m x n sparse graph
%   - 2:    m x n full
%

if isempty(opts.evalfunctor)

    switch opts.countrule
        case {'nm', 'nmx'}
            Vform = 0;
            usecvalue = strcmp(opts.countrule, 'nmx');
            fhmetric = get_metricfunc(opts.metric);
            fvote = @(x, y) evaldist_one(x, y, usecvalue, opts.clsmethod, fhmetric);

        case {'mmc', 'mms', 'mmns'}
            usecvalue = ~strcmp(opts.countrule, 'mmc');
            fhmetric = get_metricfunc(opts.metric);
            if isempty(opts.nnparams)
                Vform = 2;
                fvote = @(x, y) evaldist_all(x, y, usecvalue, fhmetric);
            else
                Vform = 1;
                fvote = @(x, y) evaldist_multi(x, y, usecvalue, opts.nnparams, fhmetric);
            end

        otherwise
            error('sltoolbox:invalidarg', ...
                'Unknown counting rule: %s', opts.countrule);
    end    

    if usecvalue
        if isempty(opts.cfunc)
            error('sltoolbox:invalidarg', ...
                'In current rule, the confidence function is required.');
        end
        if ischar(opts.cfunc) && strcmp(opts.cfunc, 'um')
            cvf = [];
        elseif isa(opts.cfunc, 'function_handle')
            cvf = opts.cfunc;
        else
            error('sltoolbox:invalidarg', 'Illegal form of cfunc');
        end
    else
        cvf = [];
    end

    evalfunctor = {@do_vote, fvote, cvf, Vform};
    
else
    evalfunctor = opts.evalfunctor;
end
        
% Do voting 
H = slvote(X0, m, X, n, evalfunctor, opts.countrule, ...
    'weights', opts.weights, ...
    'normalized', opts.normalized);


%% Core functions

function V = do_vote(X0, X, fvote, cvf, Vform)

V = fvote(X0, X);
if ~isempty(cvf)
    if Vform == 0
        V(2,:) = cvf(V(2,:));
    else
        V = cvf(V);
    end
end


function V = evaldist_one(X0, X, usecvalue, clsmethod, fhmetric)

switch clsmethod
    case 'pwcomp'
        D = fhmetric(X0, X);
        [vals, si] = min(D, [], 1);
    case 'kdtree'
        [si, vals] = annsearch(X0, X, 1);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid clsmethod: %s', clsmethod);
end

if ~usecvalue
    V = si;    
else
    V = [si; vals];
end


function V = evaldist_multi(X0, X, usecvalue, nnparams, fhmetric)
       
switch nnparams{1}
    case {'knn', 'eps'}
        use_nnparams = [nnparams, {'metric', fhmetric}];
    case 'ann'
        use_nnparams = nnparams;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid neighborhood finding method: %s', nnparams{1});
end

if usecvalue
    valtype = 'numeric';
else
    valtype = 'logical';
end

V = slnngraph(X0, X, use_nnparams, 'sparse', true, 'valtype', valtype);


function V = evaldist_all(X0, X, usecvalue, fhmetric)

V = fhmetric(X0, X);

if ~usecvalue
    V = (V ~= 0);
end



%% Auxiliary functions

function fh = get_metricfunc(m)

if ischar(m)
    fh = @(X, Y) slmetric_pw(X, Y, m);
elseif iscell(m)
    fh = @(X, Y) slmetric_pw(X, Y, m{:});
elseif isa(m, 'function_handle')
    fh = m;
else
    error('sltoolbox:invalidarg', 'The metric is specified incorrectly');
end





