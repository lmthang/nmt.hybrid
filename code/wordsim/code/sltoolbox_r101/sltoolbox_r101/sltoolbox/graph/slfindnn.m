function [nnidx, dists] = slfindnn(X0, X, method, varargin)
%SLFINDNN Finds the nearest neighbors using specified strategy
%
% $ Syntax $
%   - [nnidx, dists] = slfindnn(X0, X, method, ...)
% 
% $ Arguments $
%   - X0:           The referenced samples in which the neighbors are found
%   - X:            The query samples
%   - method:       The method to find nearest neighbors
%   - nnidx:        The indices of the nearest neighbors
%   - dists:        The distances between the samples and corresponding
%                   neighbors
%
% $ Description $
%   - [nnidx, dists] = slfindnn(X0, X, method, ...) finds the nearest
%     neighbors for all samples using the specified method. You can specify
%     the X0 and X in three different configurations:
%       - X0, []:   finds the nearest neighbors for the samples in X0, 
%                   each sample itself is not considered as a neighbor
%       - X0, X0:   finds the nearest neighbors for the samples in X0,
%                   each sample itself is also taken as a neighbor
%       - X0, X:    the query samples and the reference samples are not
%                   in the same set.
%     If there are n query samples, then nnidx is a cell array of size
%     1 x n, and each cell contains a column vector of all indices of the
%     neighbors of the corresponding sample. dists will be in the same form
%     except that the values are distances instead of indices.
%     \*
%     \t      Table. The methods for nearest neighbor finding
%     \h        name    &           description
%              'knn'    & Strict KNN using exhaustive search, having the
%                         following properties:
%                           - 'K':  The number of neighbors to find for
%                                   each query sample (default = 3)
%                           - 'maxblk': The maximum number of distances
%                                       that can be computed in one batch
%                                       (default = 1e7)
%                           - 'metric': The metric type used to compute
%                                       distances. It can be string of
%                                       the metric name, or a cell array
%                                       of parameters for slmetric_pw.
%                                       or a function handle in the form:
%                                          D = f(X1, X2)
%                                       (default = 'eucdist')
%              'ann'    & Approximate KNN using KD-tree, having the 
%                         following properties:
%                           - 'K':  The number of neighbors to find for
%                                   each query sample (default = 3)
%              'eps'    & Find all neighbors with distance below a
%                         threshold, having the following properties:
%                           - 'e':  The threshold of the distance
%                                   (default = 1)
%                           - 'maxblk': The maximum number of distances
%                                       that can be computed in one batch
%                                       (default = 1e7)
%                           - 'metric': The metric type used to compute
%                                       distances. It can be string of
%                                       the metric name, or a cell array
%                                       of parameters for slmetric_pw.
%                                       or a function handle in the form:
%                                          D = f(X1, X2)
%                                       (default = 'eucdist')
%     \*
%
% $ Remarks $
%   - In current version, the distances metric should have the attribute
%     that it decreases when the samples become nearer. Don't use 
%     similarity metrics. The metric customization only applies to 
%     'knn' and 'eps', for 'ann', it can only use Euclidean distances.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%   - Modified by Dahua Lin, on Sep 18, 2006
%       - add the functionality to support various distance metric types
%         and user-supplied distances.
%

%% parse and verify input

if nargin < 3
    raise_lackinput('slfindnn', 3);
end

if ~ismember(method, {'knn', 'ann', 'eps'})
    error('sltoolbox:invalidarg', ...
        'Invalid method for nearest neighbor finding: %s', method);
end

if isempty(X)
    X = X0;
    excludediag = true;
else
    excludediag = false;
end


%% Main skeleton

switch method
    case 'knn'
        if nargout < 2
            nnidx = find_knn(X0, X, excludediag, varargin{:});
        else
            [nnidx, dists] = find_knn(X0, X, excludediag, varargin{:});
        end
    case 'ann'
        if nargout < 2
            nnidx = find_ann(X0, X, excludediag, varargin{:});
        else
            [nnidx, dists] = find_ann(X0, X, excludediag, varargin{:});
        end
    case 'eps'
        if nargout < 2
            nnidx = find_eps(X0, X, excludediag, varargin{:});
        else
            [nnidx, dists] = find_eps(X0, X, excludediag, varargin{:});
        end
end

%% Core functions

function [nnidx, dists] = find_knn(X0, X, excludediag, varargin)

% parse input
opts.K = 3;
opts.maxblk = 1e7;
opts.metric = 'eucdist';
opts = slparseprops(opts, varargin{:});
fhmetric = get_metricfunc(opts.metric);

n = size(X, 2);
K = getK(opts, X0);
[secs, nsecs] = getparsecs(opts, X0, X);

to_output_dist = (nargout >= 2);

% prepare storage
nnidx = zeros(K, n);
if to_output_dist
    dists = zeros(K, n);
end

% compute and select
for k = 1 : nsecs
    
    % compute distances
    sp = secs.sinds(k); ep = secs.einds(k);
    curdists = compute_pwdists(X0, X, fhmetric, sp, ep, excludediag);
    
    % sort distances
    [curdists, curnnidx] = sort(curdists, 1);
    
    % selecte and record
    curnnidx = curnnidx(1:K, :);
    nnidx(:, sp:ep) = curnnidx;    
    if to_output_dist
        curdists = curdists(1:K, :);
        dists(:, sp:ep) = curdists;
    end
    
    clear curnnidx curdists;        
    
end

% organize output
nnidx = cols_to_cells(nnidx);
if nargout >= 2
    dists = cols_to_cells(dists);
end


function [nnidx, dists] = find_ann(X0, X, excludediag, varargin)

% parse input
opts.K = 3;
opts = slparseprops(opts, varargin{:});
K = getK(opts, X0);
to_output_dist = (nargout >= 2);

if excludediag
    X = [];
end

% perform search
if ~to_output_dist
    nnidx = annsearch(X0, X, K);
else
    [nnidx, dists] = annsearch(X0, X, K);
end

% organize output
nnidx = cols_to_cells(nnidx);
if to_output_dist
    dists = cols_to_cells(dists);
end
    

function [nnidx, dists] = find_eps(X0, X, excludediag, varargin)

% parse input
opts.e = 1;
opts.maxblk = 1e7;
opts.metric = 'eucdist';
opts = slparseprops(opts, varargin{:});
fhmetric = get_metricfunc(opts.metric);
[secs, nsecs] = getparsecs(opts, X0, X);
to_output_dist = (nargout >= 2);

% prepare storage
n = size(X, 2);
nnidx = cell(1, n);
if to_output_dist
    dists = cell(1, n);
end


% compute and select
for k = 1 : nsecs
    
    % compute distances
    sp = secs.sinds(k); ep = secs.einds(k);
    curdists = compute_pwdists(X0, X, fhmetric, sp, ep, excludediag);
    
    % filter
    is_selected = (curdists < opts.e);
    
    % store
    nnidx(sp:ep) = select_output_indices(is_selected);
    if to_output_dist
        dists(sp:ep) = select_output_values(curdists, is_selected);
    end
    
end



%% Auxiliary function

function dists = compute_pwdists(X0, X, fhmetric, sp, ep, excludediag)

n0 = size(X0, 2);
n = size(X, 2);

if sp == 1 && ep == n
    curX = X;
else
    curX = X(:, sp:ep);
end

% dists = slmetric_pw(X0, curX, 'eucdist');
dists = fhmetric(X0, curX);

if excludediag
    curn = ep - sp + 1;
    inds_diag = sub2ind([n0, curn], sp:ep, 1:curn);
    dists(inds_diag) = inf;
end


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


    
function C = cols_to_cells(M)   

[m, n] = size(M);
C = mat2cell(M, m, ones(1, n));


function nnidx = select_output_indices(is_selected)

n = size(is_selected, 2);
nnidx = cell(1, n);
for i = 1 : n
    nnidx{i} = find(is_selected(:,i));
end

function vals = select_output_values(vals0, is_selected)

n = size(is_selected, 2);
vals = cell(1, n);
for i = 1 : n
    vals{i} = vals0(is_selected(:,i), i);
end


function K = getK(opts, X0)

K = opts.K;
n0 = size(X0, 2);
if K >= n0
    error('sltoolbox:invalidarg', ...
        'The specified K should be less than the number of referenced samples');
end

function [secs, nsecs] = getparsecs(opts, X0, X)

n0 = size(X0, 2);
ss = max(floor(opts.maxblk / n0), 1);
n = size(X, 2);
secs = slpartition(n, 'maxblksize', ss);
nsecs = length(secs.sinds);





 
