function S = slpca(X, varargin)
%SLPCA Learns a PCA model from training samples
%
% $ Syntax $
%   - S = slpca(X)
%   - S = slpca(X, ...)
%
% $ Arguments $
%   - X:        the training sample matrix
%   - S:        the struct representing the learned PCA
%
% $ Description $
%   - S = slpca(X) learns a PCA model from the samples X by default way.
%
%   - S = slpca(X, ...) learns a PCA model from the samples X according to
%     the properties specified:
%     \*
%     \t   Table 1. The properties of PCA learning
%     \h    name       &    description                               \\
%          'method'    & The method using in training the PCA model.
%                        Currently, there are three methods available:
%                        default = 'auto'.
%                        1. 'auto': automatically selection of the best;
%                        2. 'std':  use standard way based on covariance;
%                        3. 'svd':  use SVD-based computation
%                        4. 'trans': use a transposed way, it is typically
%                           used for small-sample-size and high-dimension
%                           cases.                                    \\
%          'preserve'  & Determine how many components are preserved, it is
%                        given in following form: {sch, ...}, which is used
%                        as parameters in sldim_by_eigval.       \\
%          'weights'   & The 1 x n row vector of sample weights.  
%                        If the weights are not specified, then it 
%                        considers that all samples have weights 1. 
%                        default = [].   \\
%     \*
%                           
% $ History $
%   - Created by Dahua Lin on Apr 24, 2006
%   - Modified by Dahua Lin on Apr 25, 2006
%       - Extract the dimension determination part to an independent
%         function, in order to offer more flexible preservation process,
%         and make it reusable in more functions.
%   - Modified by Dahua Lin on Aug 17, 2006
%       - Add a field energyratio
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

% for size
if ndims(X) ~= 2
    error('sltoolbox:invaliddims', 'X should be a 2D sample matrix');
end
[d, n] = size(X);

% for options
opts.method = 'auto';
opts.preserve = {'rank'};
opts.weights = [];
opts = slparseprops(opts, varargin{:});

% check options

switch opts.method
    case 'auto'
        fh_learnpca = @pca_auto;
    case 'std'
        fh_learnpca = @pca_std;
    case 'svd'
        fh_learnpca = @pca_svd;
    case 'trans'
        fh_learnpca = @pca_trans;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid PCA learning method %s', opts.method);
end


if ~isempty(opts.weights)
    if ~isequal(size(opts.weights), [1, n])
        error('sltoolbox:sizmismatch', ...
            'The sample weights should be a 1 x n row vector');
    end
end


%% Compute

% data centralization
vmean = slmean(X, opts.weights);
X = sladdvec(X, -vmean, 1);

% sample scaling 
% in order to normalize the energy per unit sample weight
if isempty(opts.weights)
    X = sqrt(1 / n) * X;
else
    w = max(opts.weights, 0);
    tw = sum(w);
    sf = sqrt(w / tw);
    X = slmulvec(X, sf, 2);
    clear sf;
end
    
% learn full size PCA

[P, evals] = fh_learnpca(X);


% preserve principal components

evals = max(evals, 0);
kmax = min([size(P, 2), d, n-1]);
if kmax < size(P, 2)
    P = P(:, 1:kmax);
    evals = evals(1:kmax);
end
k = sldim_by_eigval(evals, opts.preserve{:});


%% Output the PCA struct

S.sampledim = d;
S.feadim    = k;

if isempty(opts.weights)
    S.support = n;
else
    S.support = tw;
end

S.vmean = vmean;
if k < kmax
    S.P = P(:, 1:k);
    S.eigvals = evals(1:k);
    S.residue = sum(evals(k+1:kmax));
else
    S.P = P;
    S.eigvals = evals;
    S.residue = 0;
end

prinenergy = sum(S.eigvals);
S.energyratio = prinenergy / (prinenergy + S.residue);


%% Sub functions for Learning PCA from centralized samples

function [P, evals] = pca_auto(X)

[d, n] = size(X);
if d <= n
    [P, evals] = pca_std(X);
else
    [P, evals] = pca_trans(X);
end


function [P, evals] = pca_std(X)

C = X * X';
[evals, P] = slsymeig(C);


function [P, evals] = pca_svd(X)

[P, D, V] = svd(X, 0);
clear V;
evals = diag(D) .^ 2;

slignorevars(V);

function [P, evals] = pca_trans(X)

Ct = X' * X;
[evals, P] = slsymeig(Ct);
P = slnormalize(X * P);




