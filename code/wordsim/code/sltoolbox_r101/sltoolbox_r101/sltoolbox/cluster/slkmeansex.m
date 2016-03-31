function [centers, labels, info] = slkmeansex(X, n, estfunctor, clsfunctor, varargin)
%SLKMEANSEX Performs Generalized K-means
%
% $ Syntax $
%   - [centers, labels] = slkmeansex(X, n, estfunctor, clsfunctor, ...)
%   - [centers, labels, info] = slkmeansex(X, n, estfunctor, clsfunctor, ...)
%
% $ Arguments $
%   - X:            the samples to be clustered
%   - n:            the number of samples
%   - estfunctor:   the functor to estimate means(centers), as follows:
%                   centers = estfunc(centers, X, K, weights, labels, ...)
%                   when input centers is empty, it performs initial
%                   estimation, otherwise, it performs updating. 
%                   In addition, it should ignore the samples with 
%                   labels being zeros or negative numbers.
%   - clsfunctor:   the functor to classify samples
%                   labels = clsfunc(centers, X, n, ...)  
%                   it should produce 1 x n row vector.
%   - centers:      the clustered centers
%   - labels:       the labels indicating which sample belong to which center
%                   a 1 x n row vector.
%   - info:         the information on iteration process
%
% $ Description $
%   - [centers, labels] = slkmeansex(X, n, estfunctor, clsfunctor, ...) 
%     is a generalized version of K-means. It actually implements an
%     iterative process to estimate centers from clustered samples and
%     re-clustered the samples according to centers.
%     You can specify the following properties:
%       - 'K':              the number of initial number of clusters
%                           (default = 3)
%       - 'init_centers':   the initial centers.
%       - 'maxiter':        the maximum number of iterations
%                           (default = 100);
%       - 'annthres':       the threshold of annealing
%                           when the sum of sample weights for a center
%                           is below annthres * the total weight, the
%                           center will be discarded. (default = 0)
%       - 'annfunc':        the function to discard a set of centers
%                           centers = annfunc(centers, inds_discard);
%       - 'weights':        the weights of the samples (default = [])
%       - 'verbose':        whether to show progress information
%                           (default = true)
%
% $ Remarks $
%   - The X and centers can be in any form that conform to the specified
%     functors.
%
%   - If init_centers is specified, K should be exactly the number of
%     initial centers.
%
%   - If annthres is 0, then no centers will be discarded even some centers
%     have no support samples in the process. The estfunctor should keep
%     those centers unchanged.
%
% $ History $
%   - Created by Dahua Lin, on Aug 28, 2006
%   - Modified by Dahua Lin, on Aug 30, 2006
%       - utilize slevalfunctor and slsharedisp
%   - Modified by Dahua Lin, on Aug 31, 2006
%       - based on slreevallearn
%

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('slkmeansex', 4);
end

opts.K = 3;
opts.init_centers = [];
opts.maxiter = 100;
opts.annthres = 0;
opts.annfunc = [];
opts.weights = [];
opts.verbose = true;
opts = slparseprops(opts, varargin{:});

if opts.K > n
    error('sltoolbox:rterror', ...
        'The initial K is larger than the number of samples');
end

if opts.annthres > 0
    if isempty(opts.annfunc)
        error('sltoolbox:invalidarg', ...
            'You should specify annfunc when annthres > 0');
    end
end

w = opts.weights;
if ~isempty(w)
    if ~isequal(w, [1 n])
        error('sltoolbox:sizmismatch', ...
            'The weights should be a 1 x n row vector');
    end
end


%% Initialization

slsharedisp_attach('slkmeansex', 'show', opts.verbose);

slsharedisp('Intialize K-Means');

if isempty(opts.init_centers)
    initcinds = randsample(n, opts.K);
    labels = zeros(1, n);
    labels(initcinds) = 1:opts.K;
    
    K = opts.K;
    centers = slevalfunctor(estfunctor, [], X, K, w, labels);
else
    K = opts.K;
    centers = opts.init_centers;
end

slsharedisp_incindent;
slsharedisp('initial K = %d', K);
slsharedisp_decindent;

labels = slevalfunctor(clsfunctor, centers, X, n);


%% Updating

slsharedisp('Update K-Means');
slsharedisp_incindent;

km_estfunctor = {@kmeansex_est, estfunctor, opts};
km_evalfunctor = {@kmeansex_eval, clsfunctor};
km_cmpfunctor = {@kmeansex_cmp};

models = {centers, K};
data = {X, n, w};
[models, labels, info] = slreevallearn(models, labels, data, ...
    km_estfunctor, km_evalfunctor, km_cmpfunctor, ...
    'iter', {'maxiter', opts.maxiter, 'titlebreak', false}, 'isrecorded', false);

centers = models{1};

slsharedisp_decindent;
slsharedisp_detach;

%% Core functions

% models = {centers, K}
% data = {X, n, w}

function models = kmeansex_est(models, data, labels, estfunctor, opts)

X = data{1};
w = data{3};
centers = models{1};
K = models{2};

if ~isempty(centers) && opts.annthres > 0    
    if isempty(w)
        w = ones(1, length(labels));
    end
    cw = sllabeledsum(w, labels, 1:K);
    wthres = opts.annthres * sum(cw) / K;
    if any(cw < wthres)
        inds_ann = find(cw < wthres);
        centers = feval(opts.annfunc, centers, inds_ann);
        K = K - length(inds_ann);
        
        models = {centers, K};
        return;
    end
end

centers = slevalfunctor(estfunctor, centers, X, K, w, labels);
models = {centers, K};


function labels = kmeansex_eval(models, data, labels, clsfunctor)

X = data{1};
n = data{2};
centers = models{1};

slignorevars(labels);
    
labels = slevalfunctor(clsfunctor, centers, X, n);


function isconverged = kmeansex_cmp(models_prev, models, labels_prev, labels)
    
K_prev = models_prev{2};
K = models{2};
n = length(labels);

slsharedisp_attach('kmeansex_cmp');

isconverged = false;
if K == K_prev
    nchanged = sum(labels ~= labels_prev);
    slsharedisp('K = %d: %d / %d changed', K, nchanged, n);

    if nchanged == 0
        isconverged = true;
    end
else
    slsharedisp('K = %d ==> %d', K_prev, K);
end

slsharedisp_detach();






    
    
    
    
        
    
    
    






