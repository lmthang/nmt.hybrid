function [Y, spectrum] = slgembed(G, Gc, d, fm, varargin)
%SLGEMBED Solves the general graph-based embedding 
%
% $ Syntax $
%   - Y = slgembed(G, Gc, d, fm, ...)
%   - [Y, spectrum] = slgembed(G, Gc, d, fm, ...)
%
% $ Arguments $
%   - G:        The graph to be optimized
%   - Gc:       The constraint graph
%   - d:        The dimension of the embedding
%   - fm:       The type of formulation
%   - Y:        The embedding sample coordinates
%
% $ Description $
%   - Y = slgembed(G, Gc, d, fm, ...) solves the general graph-based 
%     embedding of dimension d. In mathematics, it is to solve the
%     following optimization problem:
%           min/max y^T M y,  s.t. y^T C y = I
%     Based on different fm, the formulations of M and C are different:
%     fm is a cell array of two char string elements:
%           fm = {fg, fc}
%     fg indicates the formulation of M, fc indicates the formulation of
%     the constraint matrix C.
%       fg has the following different values:
%           - 'minW':   minimization using M = W, (W is the adjmat of G)
%           - 'maxW':   maximization using M = W
%           - 'minL':   minimization using M = L = D - W
%           - 'maxL':   maximization using M = L = D - W
%       fc has the following different values:
%           - 'I':      use C = I, that is y^T * y = I
%           - 'D':      use C = D, that is y^T * D * y = I (based on G)
%           - 'WC':      use C = W, adjacency matrix (based on Gc)
%           - 'LC':      use C = L = D - W, (based on Gc)
%     For example, if you specify fm = {'maxW', 'D'}, then the function 
%     will solve the following optimization problem:
%           maximize y^T W y,   s.t. y^T * D * y = I
%     You can also specify the following properties:
%       - 'inv':        The parameters do to eigenvalue inverse
%                       {method, ...}
%                       (refer to slinvevals for details)
%                       This parameter take effects only when fc = 'WC" or
%                       fc = 'LC'.
%       - 'skip':       How many eigen-components to skip. default = 0.
%                       (In some algorithms, it is necessary to skip
%                        the first or first several eigen-components).
%
%   - [Y, spectrum] = slgembed(G, Gc, d, fm, ...) additionally return
%     the spectrum of the embedding, the eigenvalues of the whitened
%     C^(-1/2) M C^(-1/2).
%     
% $ Remarks $
%   - The fc can be 'WC' or 'LC' only when Gc is full matrix, in current
%     version of implementation.
%
% $ History $
%   - Created by Dahua Lin, on Sep 12, 2006
%

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('slgembed', 4);
end

gi = slgraphinfo(G, {'square'});
n = gi.n;
W = sladjmat(G, 'valtype', 'numeric', 'sparse', issparse(G));

[fg, fc] = deal(fm{:});

Wc = [];
if strcmp(fc, 'WC') || strcmp(fc, 'LC')
    if ~isempty(Gc)
        if ~isnumeric(Gc) || issparse(Gc) || ~isequal(size(Gc), [n n])
            error('sltoolbox:invalidarg', ...
                'In current implementation, Gc must be an n x n full numeric matrix for fc = WC or GC');
        end
        Wc = Gc;
    else
        error('sltoolbox:invalidarg', ...
            'When fc is WC or LC, Wc should be specified');
    end
end

opts.inv = {};
opts.skip = 0;
opts = slparseprops(opts, varargin{:});

if d + opts.skip > n
    error('sltoolbox:invalidarg', ...
        'The embedding dimension d plus the skip dimension should not exceed n');
end


%% main skeleton

W = W + W';
if ~isempty(Wc)
    Wc = Wc + Wc';
end

% parse formulation and decide scheme

switch fg
    case 'minW'
        Mfunc = @make_Wmat;
        optype = 'max';
    case 'maxW'
        Mfunc = @make_Wmat;
        optype = 'max';
    case 'minL'
        Mfunc = @make_Lmat;
        optype = 'min';
    case 'maxL'
        Mfunc = @make_Lmat;
        optype = 'max';
    otherwise
        error('sltoolbox:invalidarg', 'Invalid fg type: %s', fg);
end

switch fc
    case 'I'
        whfunc = [];
    case 'D'
        whfunc = @(M) whM_by_D(M, W);
        nyfunc = @(Y, T) nY_by_S(Y, T);
    case 'WC'
        whfunc = @(M) whM_by_W(M, Wc, opts.inv);
        nyfunc = @(Y, T) nY_by_T(Y, T);
    case 'LC'
        whfunc = @(M) whM_by_L(M, Wc, opts.inv);
        nyfunc = @(Y, T) nY_by_T(Y, T);
    otherwise
        error('sltoolbox:invalidarg', 'Invalid fc type: %s', fc);
end


% make M
M = Mfunc(W);

% whiten M
if ~isempty(whfunc)
    MTcell = whfunc(M);
    M = MTcell{1};
    T = MTcell{2};
    clear MTcell;
end

% optimize to solve embedding
[Y, spectrum] = optim_embed(M, optype, d, opts.skip);

% normalize the embedding to satisfy the constraint
if ~isempty(whfunc)
    Y = nyfunc(Y, T);
end
Y = Y';
    

%% Core routines

function M = make_Wmat(W)

M = W;

function vD = make_Dvec(W)

vD = sum(W, 1)';
if issparse(vD)
    vD = full(vD);
end

function L = make_Lmat(W)

vD = make_Dvec(W);
n = size(W, 1);
if issparse(W)
    D = sparse((1:n)', (1:n)', vD, n, n, n);
    L = D - W;
else
    L = -W;
    dinds = (1:n)'*(n+1)-n;
    L(dinds) = L(dinds) + vD;
end

function MTcell = whM_by_D(M, W)

vD = make_Dvec(W);
vD(vD < eps) = eps;
cv = 1 ./ sqrt(vD);

n = size(M, 1);
if issparse(M)    
    Mw = M;
    for i = 1 : n
        Mw(:,i) = Mw(:,i) * cv(i);
    end
    for i = 1 : n
        Mw(i,:) = Mw(i,:) * cv(i);
    end
else
    Mw = slmulrowcols(M, cv', cv);
end

MTcell = {Mw, cv};


function MTcell = whM_by_W(M, W, invparams)

T = slwhiten_from_cov(W, invparams{:});
Mw = T' * M * T;
MTcell = {Mw, T};


function MTcell = whM_by_L(M, W, invparams)

L = make_Lmat(W);
MTcell = whM_by_W(M, L, invparams);


function [Y, spectrum] = optim_embed(M, optype, d, dskip)

switch optype
    case 'min'
        ord = 'ascend';
    case 'max'
        ord = 'descend';
end

d0 = d + dskip;
[spectrum, Y] = slsymeig(M, d0, ord);
if dskip > 0
    spectrum = spectrum(dskip+1:d0);
    Y = Y(:, dskip+1:d0);
end

function Y = nY_by_S(Y, T)

Y = slmulvec(Y, T, 1);

function Y = nY_by_T(Y, T)

Y = T * Y;



