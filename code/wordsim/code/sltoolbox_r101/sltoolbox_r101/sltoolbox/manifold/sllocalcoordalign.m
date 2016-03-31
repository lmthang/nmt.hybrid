function [GC, spectrum, LT] = sllocalcoordalign(GM, LC, dg)
%SLLOCALCOORDALIGN Performs optimal local coordinate alignment
%
% $ Syntax $
%   - [GC, spectrum, LT] = sllocalcoordalign(GM, LC)
%   - [GC, spectrum, LT] = sllocalcoordalign(GM, LC, dg)
%   - [GC, spectrum] = sllocalcoordalign(...)
%
% $ Arguments $
%   - GM:       The index map graph (n x n)
%   - LC:       The matrix of all local coordinates (dl x nnz)
%   - GC:       The global coordinates in the embedding  (dg x n)
%   - spectrum: The eigenvalues of the embedding dimensions
%   - LT:       The local transforms of all local models (dl x dg x n)
%
% $ Description $
%   - [GC, spectrum, LT] = sllocalcoordalign(GM, LC) performs optimal 
%     coordinate alignment in terms of minimizing L2 reconstruction error. 
%     This process will simultaneously resolves the optimal configuration 
%     of global coordinates in the embedded space and learns the linear
%     transforms for all sets of local coordinates to the global 
%     coordinate. By default the dimension of the global embedding will be
%     set to the same as the local dimension dl.
%     The GM is a neighborhood graph, with the neighborhood of each target
%     sample represented by the source nodes of the corresponding target
%     node. The value of the edges give the index of columns of the local
%     coordinates in LC. That is to say, LC stores all local coordinate
%     vectors obtained by applying each target model to its neighboring
%     samples and they are sorted according to the order of elements in 
%     GM. (GM should be a valued adjacency matrix)
%
%   - [GC, spectrum, LT] = sllocalcoordalign(GM, LC, dg) performs local 
%     coordinate alignment to a global embedding of the specified dimension 
%     dg.
%
%   - [GC, spectrum] = sllocalcoordalign(...) only pursues the global 
%     embedding coordinate without the need of learning local transforms.
%
% $ History $
%   - Created by Dahua Lin, on Sep 14, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sllocalcoordalign', 2);
end

gi = slgraphinfo(GM, {'adjmat', 'square', 'numeric'});
n = gi.n;

if ~isnumeric(LC) || ndims(LC) ~= 2
    error('sltoolbox:invalidarg', ...
        'LC should be a 2D numeric matrix');
end
[dl, ny] = size(LC);
if ny ~= nnz(GM)
    error('sltoolbox:invalidarg', ...
        'The number of samples in LC does not match the non-zero entries in GM');
end

if nargin < 3
    dg = dl;
end

if nargout >= 3
    learnLT = true;
else
    learnLT = false;
end


%% Pursue optimal embedding (global coordinates)

% prepare data structure
B = prepareBmat(GM, n);
if learnLT
    LRs = cell(n, 1);
end

% construct W matrices

for i = 1 : n
    
    % get local coords
    cinds = find(GM(:,i));
    curlc = LC(:, GM(cinds, i));
    
    % decompose
    cn = size(curlc, 2);
    [cu, cs, cv] = svd(curlc, 0);
    
    %decide rank
    cs = diag(cs);
    rk = max(sum(cs > eps(cs(1)) * cn), 1);
    
    % compute W, M and add it to B
    if rk < cn              % note that when rk == cn, the contribution of current model is zero 
        cv = cv(:, 1:rk);
        W = eye(cn) - cv * cv';     % note that for W = I - VV^T, it always has that W = W * W
        vm = sum(W, 1) / cn;
        M = sladdrowcols(W, -vm, -vm') + sum(vm) / cn; 
        B(cinds, cinds) = B(cinds, cinds) + M;                
    end
    
    % prepare for learning local transforms (make use of u, s, v, so that
    % we need not to do svd again
    if learnLT
        cLTR = compute_pinv(cu(:, 1:rk), cs(1:rk), cv);
        LRs{i} = sladdvec(cLTR, -sum(cLTR, 1)/cn);
    end
    
end

% post process B
B = postprocessBmat(B);

% solve eigen-problem of B
[spectrum, GC] = slsymeig(B, dg+1, 'ascend');
spectrum = spectrum(2:dg+1);
GC = GC(:, 2:dg+1)';


%% Learn the local transforms
LT = zeros(dl, dg, n);

if learnLT
    for i = 1 : n
        cLR = LRs{i};            
        cGC = GC(:, GM(:,i) ~= 0);
        cLT = cGC * cLR;
        LT(:,:,i) = cLT;
    end    
end




%% Auxiliary functions

function B = prepareBmat(GM, n)
% estimate the number of non-zeros in B and prepare the most efficient
% storage.
% The strategy is
%   first allocate a logical array to record which elements may possibly
%   be set to zeros
% Then according to nnz, 
%   if nnz > n * n / 4, use full matrix, make an n x n zeros
%   otherwise use sparse matrix, make with all elements that would be used
%   set to 1 first. After all computation, these ones should be reduced
%   using postprocessBmat
%

% estimate the maximum number of non-zeros
nnzb = 0;
for i = 1 : n
    cnnb = nnz(GM(:,i));
    nnzb = nnzb + cnnb * cnnb;
end
nnzb = min(nnzb, n * n);

% prepare the indicator matrix
if n * n > nnzb * 20
    Z = sparse(1, 1, false, n, n, nnzb);
else
    Z = false(n, n);
end

% set the indicators
for i = 1 : n
    cinds = find(GM(:,i));
    Z(cinds, cinds) = 1;
end

% re-estimate the nnz accurately
nnzb = nnz(Z);

% make B
if n * n > nnzb * 4
    B = zeros(n, n);
else
    [I, J] = find(Z);    
    B = sparse(I, J, 1, n, n);
end


function B = postprocessBmat(B)

if issparse(B)
    B = spfun(@(x) x - 1, B);
end


function R = compute_pinv(u, s, v)

R = v * diag(1 ./ s) * u';

    





    
    



    
    







