function T = slgbfe(X, G, Gc, dy, fm, varargin)
%SLGBFE Performs Graph-based Feature Extraction Learning
%
% $ Syntax $
%   - T = slgbfe(X, G, Gc, dy, fm, ...)
%
% $ Arguments $
%   - X:        The sample matrix 
%   - G:        The graph to be optimized
%   - Gc:       The constraint graph
%   - dy:       The dimension of feature space
%   - fm:       The formulation type
%   - T:        The learned transform matrix (dx x dy)
%               the transform is done by y = T' * x
%
% $ Description $
%   - T = slgbfe(X, G, Gc, dy, fm, ...) performs graph-based feature 
%     extraction learning. It is to solve the following optimization.
%       
%       min/max  T'X M(G) X'T,    s.t. T'X M(Gc) X'T = I
%
%     The concrete formulation depends on the formulation type given in
%     fm = {fg, fc}. For fg, it has the following three types:
%       - 'minW':   do minimization with M(G) = W
%       - 'maxW':   do maximization with M(G) = W
%       - 'minL':   do minimization with M(G) = L = D - W
%       - 'maxL':   do maximization with M(G) = L = D - W
%     For fc, it has the following three types:
%       - 'O':      constraint T be orthogonal: T'*T = I (ignore Gc)
%       - 'I':      constraint T'* X * X' * T = I (ignore Gc)
%       - 'WC':     constraint with M(Gc) = W of Gc
%       - 'LC':     constraint with M(Gc) = L of Gc: D - W
%     In the aforementioned formulation, W is the adjacency matrix, while
%     L is the Laplacian matrix. When Gc is ignored (as in 'O' and 'I'),
%     you can just input Gc as [].
%
%     You can further specify the following properties to control the 
%     learning process:
%       - 'whparams':  The parameters for doing whitening of M(Gc), please
%                      refer to the function slwhiten_from_cov. The params 
%                      are given in a cell array as {method, ...}. 
%                      default = {}
%       - 'skip':      The number of components to be skipped. default = 0
%
% $ Remarks $
%   - The implementation is based on slgembed.
%
%   - The function will not centralize the samples, if it is needed please
%     centralize them before invoking.
%
% $ History $
%   - Created by Dahua Lin, on Sep 17, 2006
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slgbfe', 5);
end

if ~isnumeric(X) || ndims(X) ~= 2
    error('sltoolbox:invalidarg', 'X should be a 2D numeric matrix');
end
n = size(X, 2);

if ~iscell(fm) || length(fm) ~= 2
    error('sltoolbox:invalidarg', 'fm should be a length-2 cell array');
end
fg = fm{1};
fc = fm{2};

gi = slgraphinfo(G, {[n, n]});
W = sladjmat(G, ...
    'valtype', 'numeric', ...
    'sparse', strcmp(gi.form, 'adjmat') && issparse(G));

if strcmp(fc, 'WC') || strcmp(fc, 'LC')
    if isempty(Gc)
        error('sltoolbox:invalidarg', ...
            'When fc is WC or LC, Gc should not be empty');
    end
    slgraphinfo(Gc, {'adjmat', [n, n]});
    if isnumeric(Gc)
        Wc = Gc;
    else
        Wc = double(Gc);
    end
else
    Wc = [];
end


opts.whparams = {};
opts.skip = 0;
opts = slparseprops(opts, varargin{:});


%% Construct problem

% enforce symmetry
W = (W + W') * (1/2);
if ~isempty(Wc)
    Wc = (Wc + Wc') * (1/2);
end

% construct re-formulated G: R
switch fg
    case 'maxW'
        R = X * W * X';
        rfg = 'maxW';
    case 'minW'
        R = X * W * X';
        rfg = 'minW';
    case 'maxL'
        R = X * make_Lmat(W) * X';
        rfg = 'maxW';
    case 'minL'
        R = X * make_Lmat(W) * X';
        rfg = 'minW';
    otherwise
        error('sltoolbox:invalidarg', 'Invalid fg name: %s', fg);
end    

% construct re-formulated Gc: Rc
switch fc
    case 'O'
        Rc = [];
        rfc = 'I';
    case 'I'
        Rc = X * X';
        rfc = 'WC';
    case 'WC'
        Rc = X * Wc * X';
        rfc = 'WC';
    case 'LC'
        Rc = X * make_Lmat(Wc) * X';
        rfc = 'WC';
    otherwise
        error('sltoolbox:invalidarg', 'Invalid fc name: %s', fc);
end


%% solve problem

Y = slgembed(R, Rc, dy, {rfg, rfc}, ...
    'inv', opts.whparams, ...
    'skip', opts.skip);                 
T = Y';


%% Computational routines

function L = make_Lmat(W)

vD = sum(W, 1)';
if issparse(vD)
    vD = full(vD);
end

n = size(W, 1);
if issparse(W)
    D = sparse((1:n)', (1:n)', vD, n, n, n);
    L = D - W;
else
    L = -W;
    dinds = (1:n)'*(n+1)-n;
    L(dinds) = L(dinds) + vD;
end

