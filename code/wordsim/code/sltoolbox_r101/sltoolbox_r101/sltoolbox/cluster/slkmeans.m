function [means, labels] = slkmeans(X, varargin)
%SLKMEANS Performs K-Means Clustering on samples
%
% $ Syntax $
%   - [means, labels] = slkmeans(X, ...)
%
% $ Arguments $
%   - X:        the sample matrix
%   - means:    the center(mean) vectors of the clusters
%   - labels:   the labels of the clusters which the samples belong to
%
% $ Description $
%   - [means, labels] = slkmeans(X, ...) Performs K-Means Clustering on 
%     the data X with each column representing a sample. If k is the 
%     number of clusters. In the output argument, means are the d x k 
%     vectors representing the centers of clusters. labels indicates 
%     which cluster the elements belong to. You can specify the following
%     additional properties.
%
%     \*
%     \t  Table 1. Clustering properties
%     \h    name        &     description
%          'K'          & The number of initial clusters, default = 3.
%          'init_means' & The initial values of cluster centers. 
%                         (default = [], that is random draw)
%          'clsfunc'    & The function for classifying samples given
%                         the means of clusters. It can be one of the
%                         following string:
%                         1. 'normal' (default): use slmetric_pw for
%                            distance calculation;
%                         2. 'samplewise': classify samples one-by-one
%                         3. 'ann': classify samples using annsearch
%                            (annsearch is required)
%                         or, clsfunc can be a function handle using
%                         following syntax labels = f(centers, data).
%          'maxiter'    & The maximum number of iterations (default = 100)
%          'annthres'   & The threshold of center annealing
%                         (default = 0)
%          'weights'    & The weights of the samples
%          'verbose'    & Whether to show dynamic information in the 
%                         procedure (default = true)
%
% $ History $
%   - Created by Dahua Lin on Oct 7th, 2005
%   - Modified by Dahua Lin on Apr 24th, 2006
%       - Upgrade the function to base on sltoolbox v4
%       - Add the clsfunc properties, so that the user can customize
%         the behaviour of classification step according to the context.
%   - Modified by Dahua Lin on Aug 28, 2006
%       - Based on the new framework function slkmeansex
%       - Incorporate the support of center annealing
%   - Modified by Dahua Lin on Sep 14th, 2006
%       - use sllabelinds to increase the efficiency of gathering the
%         samples in the same cluster in the estimation step.
%      

%% parse and verify input arguments
if ndims(X) ~= 2 
    error('sltoolbox:invaliddims', 'X should be a 2D matrix');
end

opts.K = 3;
opts.init_means = [];
opts.clsfunc = 'normal';
opts.maxiter = 100;
opts.annthres = 0;
opts.weights = [];
opts.verbose = true;
opts = slparseprops(opts, varargin{:});

n = size(X, 2);

if ischar(opts.clsfunc)
    switch opts.clsfunc
        case 'normal'
            fh_classify = @classify_normal;
        case 'samplewise'
            fh_classify = @classify_samplewise;
        case 'ann'
            fh_classify = @classify_ann;
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid clsfunc option %s', opts.clsfunc);
    end
elseif isa(opts.clsfunc, 'function_handle')
    fh_classify = opts.clsfunc;
else
    error('sltoolbox:invalidarg', ...
        'clsfunc can be either a string or a function handle');
end


%% Perform K-means

estfunctor = {@kmeans_est};
clsfunctor = {@kmeans_classify, fh_classify};
annfunc = @kmeans_anneal;

[means, labels] = slkmeansex(X, n, estfunctor, clsfunctor, ...
    'K', opts.K, ...
    'init_centers', opts.init_means, ...
    'maxiter', opts.maxiter, ...
    'annthres', opts.annthres, ...
    'annfunc', annfunc, ...
    'weights', opts.weights, ...
    'verbose', opts.verbose);



%% Core slot functions

function centers = kmeans_est(centers, X, K, weights, labels)

d = size(X, 1);
if isempty(centers)
    centers = zeros(d, K);
end

Inds = sllabelinds(labels, 1:K);
for i = 1 : K    
    si = Inds{i};
    
    if ~isempty(si)
        curX = X(:, si);    
        if isempty(weights)
            curw = [];
        else
            curw = weights(si);
        end        
        centers(:, i) = slmean(curX, curw);    
    end
end


function labels = kmeans_classify(centers, X, n, fh_classify)

slignorevars(n);
labels = fh_classify(centers, X);


function centers = kmeans_anneal(centers, inds_discard)

centers(:, inds_discard) = [];




%% The functions for classifying samples to clusters

function L = classify_normal(centers, data)

dists = slmetric_pw(centers, data, 'eucdist');
[md, L] = min(dists, [], 1);
slignorevars(md);
    
function L = classify_samplewise(centers, data)

n = size(data, 2);
L = zeros(1, n);
for i = 1 : n
    curdists = slmetric_pw(centers, data(:, i), 'eucdist');
    [md, p] = min(curdists);
    L(i) = p;
end
slignorevars(md);

function L = classify_ann(centers, data)

L = annsearch(centers, data, 1);
L = L(:)';







    
    