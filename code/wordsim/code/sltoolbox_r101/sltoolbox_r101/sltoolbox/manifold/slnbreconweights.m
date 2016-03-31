function WG = slnbreconweights(X0, X, G, varargin)
%SLNBRECONWEIGHTS Solve the optimal reconstruction weights on given neighbors
%
% $ Syntax $
%   - WG = slnbreconweights(X0, X, G, ...)
%
% $ Arguments $
%   - X0:       The reference samples to reconstruct the query samples
%   - X:        The query samples
%   - G:        The graph giving the neighborhood relations 
%   - WG:       The weighted graph giving the solved weights
%
% $ Description $
%   - WG = slnbreconweights(X0, X, G, ...) solves the optimal weights to
%     reconstruct the samples in X from those in X0. If X is empty, then
%     it would use X0 as X. The graph G indicates the neighborhood 
%     relation, having n0 sources and n targets. The WG is a graph with
%     of the same size as G, and the reconstuction weights are placed
%     in the positions corresponding to those in G. 
%     You can specify the following properties to control the solving:
%     \*
%     \t    Table. The Properties of Reconstruction Weights Solving
%     \h       name        &     description
%            'constraint'  & The constraint on the solution, it can be
%                            one of the following string to indicate a
%                            single constraint or a cell array of multiple
%                            strings to indicate compound constaints.
%                            - 'nonneg':  non-negative
%                            - 's1':      the weights sum to 1
%                            (default = 's1')
%            'delta'       & The value of regularization. In practice, 
%                            regularization is essential to guarantee the
%                            stability of the solution. In implementation,
%                            the diagonal elements of the gram matrix will
%                            be added with a value:
%                               (delta^2) * trace(G) / K
%                            here G is X^T * X, K is the neighbor number.
%                            (default = 0.1)
%            'solver'      & The solver offered by user (function handle).  
%                            If the user specify a non-empty solver, then 
%                            it will use the user's solver to solve 
%                            weights. The solver is like the form:
%                               w = f(X, y)
%                            Here X is d x K neighbor sample matrix, y is
%                            a d x 1 vector representing the target sample.
%                            It should output a K x 1 vector giving the
%                            reconstruction weights. 
%                            By default, solver = [], indicating to use
%                            internal solver based on constraint and delta.   
%            'thres'       & The thres, if the ratio of a weight value
%                            to the average weight for that reconstruction
%                            is lower than thres, the weight is set 
%                            to strictly zeros. This would significantly
%                            reduces the near-zero weights, and thus
%                            reduces the complexity of the graph.
%                            (default = 1e-8)
%     \*
%
% $ Remarks $
%   - When the user specify a non-empty solver, the internal solver will
%     not be used, thus constraint and delta will not take effect.
%
%   - G would be in all acceptable graph form. WG will always be a 
%     numeric matrix. If G is a sparse adjmat, then WG would be sparse,
%     otherwise WG is full.
%
%   - With the WG solved, to reconstruct by weighed combination of
%     neighbors, you can simply write it as: Xr = X0 * WG, then Xr
%     is a d x n matrix with the j-th column reconstructed from the 
%     referenced samples in Xr using the j-th column's weights in WG.
%
% $ History $
%   - Created by Dahua Lin, on Sep 11st, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slnbreconweights', 3);
end

if ~isnumeric(X0) || ndims(X0) ~= 2
    error('sltoolbox:invalidarg', ...
        'X should be a 2D numeric matrix');
end
[d, n0] = size(X0);


if isempty(X)
    X = X0;    
else
    if ~isnumeric(X) || ndims(X) ~= 2
        error('sltoolbox:invalidarg', ...
            'X0 should be a 2D numeric matrix');
    end        
    if size(X, 1) ~= d
        error('sltoolbox:sizmismatch', ...
            'The sample dimension in X is not the same as that in X0');
    end
end
n = size(X, 2);

gi = slgraphinfo(G);
if gi.n ~= n0 || gi.nt ~= n
    error('sltoolbox:sizmismatch', ...
        'The size of the graph is not consisitent with the sample set');
end

opts.constraint = 's1';
opts.delta = 0.1;
opts.solver = [];
opts.thres = 1e-8;
opts = slparseprops(opts, varargin{:});

thres = opts.thres;

%% Prepare parameters

% prepare graph
if ~strcmp(gi.form, 'adjmat')
    G = sladjmat(G, 'sparse', true, 'valtype', 'logical');
end
    
% prepare solver
if isempty(opts.solver)
       
    % parse constraints
    cs = opts.constraint;
    if ~isempty(cs)
        if ~iscell(cs)
            cs = {cs};
        end
        constraint = parse_constraints(cs);
    else
        constraint = parse_constraints({});
    end
    
    % decide solver
    if ~constraint.nonneg
        delta2 = opts.delta^2;
        if ~constraint.s1
            wsolver = @(X, y) internal_wsolver_unc(X, y, delta2);
        else
            wsolver = @(X, y) internal_wsolver_s1(X, y, delta2);
        end
    else
        optimopts = optimset('Display', 'off', 'LargeScale', 'off');
        if ~constraint.s1
            wsolver = @(X, y) internal_wsolver_nonneg(X, y, opts.delta, optimopts);
        else
            wsolver = @(X, y) internal_wsolver_nonneg_s1(X, y, opts.delta, optimopts);
        end
    end                           
    
else
    if ~isa(opts.solver, 'function_handle')
        error('The weight solver should be a function handle');
    end
    wsolver = opts.solver;
end


%% main skeleton

% init WG
if issparse(G)
    WG = spalloc(n0, n, nnz(G));
else
    WG = zeros(n0, n);
end

% solve weights
for i = 1 : n
    nbinds = find(G(:,i));
    if ~isempty(nbinds)        
        Xnb = X0(:, nbinds);
        y = X(:,i);
        w = wsolver(Xnb, y);
        if thres > 0
            absw = abs(w);
            curthres = thres * sum(absw) / length(w);
            w(absw < curthres) = 0;
        end        
        WG(nbinds, i) = w;
    end    
end


%% constraint parsing function

function c = parse_constraints(cs)

ncs = length(cs);

c = struct('nonneg', false, 's1', false);

for i = 1 : ncs
    cname = cs{i};
    if ~ischar(cname)
        error('sltoolbox:invalidarg', ...
            'The constraint should be given in char string');
    end
    switch cname
        case 'nonneg'
            c.nonneg = true;
        case 's1'
            c.s1 = true;
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid constraint name for weight solving: %s', cname);
    end
end



%% The internal weight solvers

% unconstrained solver
function w = internal_wsolver_unc(X, y, delta2)

[G, Xty] = compute_G_Xty(X, y, delta2);
w = G \ Xty; 

% solver with s1 constraint
function w = internal_wsolver_s1(X, y, delta2)

[G, Xty, K] = compute_G_Xty(X, y, delta2);
wu = G \ [Xty, ones(K, 1)];

w = wu(:,1);
u = wu(:,2);
lambda = (1 - sum(w)) / sum(u);
w = w + lambda * u;

% solver with nonnegative constraint
function w = internal_wsolver_nonneg(X, y, delta, optimopts)

[Xa, ya, K] = augformulate(X, y, delta);
if K <= 20
    w = lsqnonneg(Xa, ya, [], optimopts);
else
    lb = zeros(K, 1);
    w = lsqlin(Xa, ya, [], [], [], [], lb, [], [], optimopts);
end

% solver with nonnegative and s1 constraint
function w = internal_wsolver_nonneg_s1(X, y, delta, optimopts)

[Xa, ya, K] = augformulate(X, y, delta);

Aeq = ones(1, K);
beq = 1;
lb = zeros(K, 1);
w = lsqlin(Xa, ya, [], [], Aeq, beq, lb, [], [], optimopts);


% solver preparation function

function [G, Xty, K] = compute_G_Xty(X, y, delta2)

% compute Xt, G, and Xty
K = size(X, 2);
Xt = X';
G = Xt * X;
Xty = Xt * y;

% regularize
if delta2 > 0
    diaginds = (1:K)*(K+1) - K;
    rv = delta2 * sum(G(diaginds)) / K;
    G(diaginds) = G(diaginds) + rv;
end

function [Xa, ya, K] = augformulate(X, y, delta)

K = size(X, 2);
if delta ~= 0
    Xa = [X; delta * eye(K)];
    ya = [y; zeros(K, 1)];
else
    Xa = X;
    ya = y;
end

