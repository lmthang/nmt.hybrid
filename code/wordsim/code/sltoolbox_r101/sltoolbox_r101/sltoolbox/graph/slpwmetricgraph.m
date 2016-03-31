function G = slpwmetricgraph(X, varargin)
%SLPWMETRICGRAPH Constructs a graph based on pairwise metrics
%
% $ Syntax $
%   - G = slpwmetricgraph(X, ...)
%   - G = slpwmetricgraph(X, Xt, ...)
%
% $ Arguments $
%   - X:        The sample matrix with each column as a (source) node
%   - Xt:       The sample matrix with each column as a (target) node  
%   - G:        The constructed graph
%   
% $ Description $
%   - G = slpwmetricgraph(X, ...) constructs a graph based on computation
%     of pairwise metric between vector samples. You can specify the 
%     following properties:
%     \*
%     \t   The Properties of Graph Matrix construction           \\
%     \h      name       &      description
%            'sparse'    & whether the target graph G is sparse 
%                          (default = true)
%            'valtype'   & The type of values in G: 'logical'|'numeric'
%                          (default = 'numeric')
%                          The value output by evalfunctor should conform
%                          to the specified valtype
%            'maxblk'    & The maximum number of elements that can be
%                          computed in each batch. (default = 1e7)   
%            'mfunctor'  & The functor to compute the metrics. In the form:
%                          V = f(X1, X2, ...)
%                          default = {@slmetric_pw, 'eucdist')
%            'tfunctor'  & The functor to transform the metric values,
%                          like tv = f(sv, ...)
%                          The transform is taken before the threshold 
%                          filtering. default = []
%            'thres'     & The threshold values. The values not in the
%                          valid range set by the threshold is regarded as
%                          zeros. 
%                          (default = [], means not to use threshold)
%            'threstype' & The type of threshold value, (default = 'ub')
%                           - 'lb': a lower bound scalar 
%                                   valid_range: x >= thres
%                           - 'ub': a upper bound scalar
%                                   valid_range: x <= thres
%                           - 'lub': a vector of lower and upper bounds
%                                    valid_range: thres(1) <= x <= thres(2)                           
%     \*
%
%   - G = slpwmetricgraph(X, Xt, ...) constructs a bigraph with different
%     source and target node set. The properties mentioned above are
%     all applicable here.
%
% $ Remarks $
%   - The implementation is based on slpwgraph.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%   - Modified by Dahua Lin, on Sep 10th, 2006
%       - Base on upgraded slpwgraph function
%       - Add support for bigraph
%

%% parse and verify input

if nargin >= 2
    if isnumeric(varargin{1})
        Xt = varargin{1};
        if nargin == 2
            params = {};
        else
            params = varargin(2:end);
        end
    else
        Xt = X;
        params = varargin;
    end
else
    Xt = X;
    params = {};
end

opts.sparse = true;
opts.valtype = 'numeric';
opts.maxblk = 1e7;
opts.mfunctor = {@slmetric_pw, 'eucdist'};
opts.tfunctor = [];
opts.thres = [];
opts.threstype = 'ub';
opts = slparseprops(opts, params{:});

if ~ismember(opts.threstype, {'lb', 'ub', 'lub'})
    error('sltoolbox:invalidarg', ...
        'Invalid type of threshold: %s', opts.threstype);
end

%% Main skeleton

n = size(X, 2);
nt = size(Xt, 2);
mfunctor = opts.mfunctor;
tfunctor = opts.tfunctor;
filter = getvaluefilter(opts);
evalfunctor = {@metricevalfunc, mfunctor, tfunctor, filter};

G = slpwgraph(X, Xt, n, nt, evalfunctor, ...
    'sparse', opts.sparse, ...
    'valtype', opts.valtype, ...
    'maxblk', opts.maxblk);


%% Core functions

function V = metricevalfunc(X, Xt, inds1, inds2, mfunctor, tfunctor, filter)

X1 = X(:, inds1);
X2 = Xt(:, inds2);
V = slevalfunctor(mfunctor, X1, X2);

if ~isempty(tfunctor)
    V = slevalfunctor(tfunctor, V);
end

if ~isempty(filter)
    zero_range = ~filter(V);
    V(zero_range) = 0;
end    


function filter = getvaluefilter(opts)

if isempty(opts.thres)
    filter = [];
else
    thr = opts.thres;
    switch opts.threstype
        case 'lb'
            if ~isscalar(thr)
                error('sltoolbox:invalidarg', 'For lb type, the thres should be a scalar');
            end
            filter = @(x) x >= thr;
        case 'ub'
            if ~isscalar(thr)
                error('sltoolbox:invalidarg', 'For ub type, the thres should be a scalar');
            end
            filter = @(x) x <= thr;
        case 'lub'
            if ~isvector(thr) || length(thr) ~= 2
                error('sltoolbox:invalidarg', 'For lub type, the thres should be a length-2 vector');
            end
            filter = @(x) x >= thr(1) & x <= thr(2);
    end
end
            
