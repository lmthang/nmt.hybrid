function [Y, spectrum] = sllemap(G, d, sch)
%SLLEMAP Solves Laplacian Eigenmap Embedding
%
% $ Syntax $
%   - Y = sllemap(G, d)
%   - Y = sllemap(G, d, sch)
%   - [Y, spectrum] = sllemap(...)
%
% $ Arguments $
%   - G:        The affinity graph (in any acceptable form): n x n
%   - d:        The embedding dimension
%   - sch:      The scheme to use
%   - Y:        The solved embedded coordinates (d x n)
%
% $ Description $
%   - Y = sllemap(G, d) uses the default scheme to solve the Laplacian
%     Eigenmap embedding in a d-dimensional space.
%
%   - Y = sllemap(G, d, sch) uses the specified scheme to solve the 
%     Laplacian Eigenmap embedding in a d-dimensional space.
%
%     Three schemes are implemented to resolve the problem, they are
%     under three different formulations:
%       (1) 'minLI':
%           objective: minimize sum_ij w_ij ||y_i - y_j||^2
%                      s.t. forall i, ||y_i||^2 = 1
%           in matrix form, it is expressed as:
%               minimize y^T * L * y,  s.t. y^T * y = 1
%       (2) 'minLD': 
%           objective: minimize sum_ij w_ij ||y_i - y_j||^2
%                      s.t. forall i, d_ii ||y_i||^2 = 1
%           in matrix form, it is expressed as:
%               minimize y^T * L * y, s.t. y^T * D * y = 1
%           This is the original formulation in many papers in the fields
%           of spectral learning, clustering and manifold learning.
%       (3) 'maxWD': (default)
%           objective: maximize sum_ij w_ij <y_i, y_j>
%                      s.t. forall i, d_ii ||y_i||^2 = 1
%           in matrix form, it is expressed as:
%               maximize y^T * W * y, s.t. y^T * D * y = 1
%           In theory, this objective is equivalent to 'minLD'. However,
%           due to that its implementation is based on finding the 
%           eigenvectors corresponding to the largest eigenvalues instead
%           of the smallest ones, thus it is much more efficient and
%           numerically stable. Hence, it is selected as the default
%           scheme.
%
%   - [Y, spectrum] = sllemap(...) additionally outputs the spectrum 
%     of the embedding. The definition of the spectrum varies for different
%     schemes:
%     'minLI': the spectrum is the eigenvalues of L, in ascending order
%     'minLD': the spectrum is the eigenvalues of D^(-1/2) * L * D^(-1/2),
%              in ascending order
%     'maxWD': the spectrum is the eigenvalues of D^(-1/2) * W * D^(-1/2),
%              in descending order.
%
% $ Remarks $
%   - If the input graph does not have edge values or it is logical, 
%     it just assume 1 between neighboring samples and 0 for other pairs.
%
%   - The embedding dimension d should be strictly less than the number
%     of samples n.
%
% $ History $
%   - Created by Dahua Lin, on Sep 12nd, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sllemap', 2);
end

gi = slgraphinfo(G, {'square'});
n = gi.n;
if strcmp(gi, 'adjmat')
    if isnumeric(G)
        W = G;
    else
        W = double(G);
    end
else
    W = sladjmat(G);
end

if d >= n
    error('sltoolbox:invalidarg', ...
        'The embeded dimension d should be strictly less than n');
end

if nargin < 3 || isempty(sch)
    sch = 'maxWD';
end

%% main delegation

% L = Di + Dj - Wij - Wji
% we let
%   W = Wij + Wji
%   D = Di + Dj = diag(diag(W))
%   L = D - W
W = W + W';

switch sch
    case 'maxWD'
        [Y, spectrum] = solve_maxWD(W, d);
    case 'minLD'
        [Y, spectrum] = solve_minLD(W, d);
    case 'minLI'
        [Y, spectrum] = solve_minLI(W, d);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid scheme for solving eigenmap: %s', sch);
end

%% Core functions

function [Y, spectrum] = solve_maxWD(W, d)

vD = calcDv(W);
W = calcNormalizeMat(W, vD);

[spectrum, Y] = slsymeig(W, d+1, 'descend');
spectrum = spectrum(2:d+1);
Y = Y(:, 2:d+1)';
Y = denormY(Y, vD);

function [Y, spectrum] = solve_minLD(W, d)

vD = calcDv(W);
L = calcL(vD, W);
L = calcNormalizeMat(L, vD);

[spectrum, Y] = slsymeig(L, d+1, 'ascend');
spectrum = spectrum(2:d+1);
Y = Y(:, 2:d+1)';
Y = denormY(Y, vD);


function [Y, spectrum] = solve_minLI(W, d)

vD = calcDv(W);
L = calcL(vD, W);

[spectrum, Y] = slsymeig(L, d+1, 'ascend');
spectrum = spectrum(2:d+1);
Y = Y(:, 2:d+1)';


%% Computation routines

function vD = calcDv(W)

vD = sum(W, 1)';

function L = calcL(vD, W)

n = length(vD);
if issparse(W)
    D = sparse((1:n)', (1:n)', vD, n, n, n);
    L = D - W;
else
    L = -W;
    dinds = (1:n)'*(n+1) - n;
    L(dinds) = L(dinds) + vD;
end

function Mn = calcNormalizeMat(M, vD)

vD(vD < eps) = eps;
cv = 1 ./ sqrt(vD);

if issparse(M)
    n = size(M,1);
    Mn = M;
    for i = 1 : n
        Mn(:,i) = Mn(:,i) * cv(i);
    end
    for i = 1 : n
        Mn(i,:) = Mn(i,:) * cv(i);
    end
else    
    rv = cv';
    Mn = slmulrowcols(M, rv, cv);
end

function Y = denormY(Y, vD)

vD = vD';
vD(vD < eps) = eps;
Y = slmulvec(Y, 1 ./ sqrt(vD), 2);




