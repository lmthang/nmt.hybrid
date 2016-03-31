function [P1, P2, spectrum] = slcopca(X1, X2, d, varargin)
%SLCOPCA Performs Coupled PCA Learning
%
% $ Syntax $
%   - [P1, P2] = slcopca(X1, X2, sch, ...)
%   - [P1, P2, spectrum] = slcopca(...)
%
% $ Arguments $
%   - X1:       The samples in the first modality
%   - X2:       The samples in the second modality
%   - d:        The target space dimension
%   - P1:       The projection matrix (bases) of the first modality space
%   - P2:       The projection matrix (bases) of the second modality space
%   - spectrum: The covariance energy along dimensions of target space
%   
% $ Description $
%   - [P1, P2] = slcopca(X1, X2, sch, ...) performs coupled PCA learning
%     for two correlated sample spaces. The learning objective is to
%     pursue two subspaces such that they are maximally correlated. The
%     objective function is formulated as
%
%       maximize trace( P1^T * C_12 * P2 * P2^T * C_21 * P1 ) / n
%           s.t. P1^T * P1 = I, and  P2^T * P2 = I
%       where C_12 is the covariance between X1 and X2, 
%             C_21 is the covariance between X2 and X1
%
%     Suppose the dimensions for the two spaces are d1 and d2 respectively, 
%     and n pairs of corresponding samples are given in X1 and X2. Then X1 
%     and X2 should be d1 x n and d2 x n matrices respectively. 
%     You can further specify the following properties to control the
%     learning procedure:
%       - 'weights':    The weights of the samples, default = []
%       - 'mean1':      The pre-computed mean vector for X1, default = []
%                       if mean1 is set as 0, then it means that X1 has 
%                       been centralized.
%       - 'mean2':      The pre-computed mean vector for X2, default = []
%       
%   - [P1, P2, spectrum] = slcopca(...) also outputs the spectrum. You
%     can specify the following properties to control the type of the
%     output spectrum:
%       - 'spectype':   The type of output spectrum
%                       - 'normal': The average energies along target
%                                   dimensions.
%                       - 'ratio':  The ratio of preserved energy along
%                                   target dimensions to the total
%                                   energy.
%                       default = 'normal'.
% 
% $ History $
%   - Created by Dahua Lin, on Sep 16, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slcopca', 3);
end

if ~isnumeric(X1) || ~isnumeric(X2) || ndims(X1) ~= 2 || ndims(X2) ~= 2
    error('sltoolbox:invalidarg', ...
        'X1 and X2 should be 2D numeric matrices');
end

[d1, n] = size(X1);
[d2, n2] = size(X2);
if n ~= n2
    error('sltoolbox:sizmismatch', ...
        'The numbers of samples in X1 and X2 do not match');
end

dmax = min(d1, d2);
if d > dmax
    error('sltoolbox:invalidarg', ...
        'The target dimension d should not exceed d1 or d2');
end

opts.weights = [];
opts.mean1 = [];
opts.mean2 = [];
opts.spectype = 'normal';
opts = slparseprops(opts, varargin{:});

w = opts.weights;
if ~isempty(w)
    if ~isequal(size(w), [1, n])
        error('sltoolbox:sizmismatch', ...
            'w should be a 1 x n row vector');
    end
end

vmean1 = opts.mean1;
vmean2 = opts.mean2;
if ~isempty(vmean1) && ~isequal(vmean1, 0) && ~isequal(size(vmean1), [d1, 1])
    error('sltoolbox:sizmismatch', ...
        'The size of mean1 is illegal');
end
if ~isempty(vmean2) && ~isequal(vmean2, 0) && ~isequal(size(vmean2), [d2, 1])
    error('sltoolbox:sizmismatch', ...
        'The size of mean1 is illegal');
end

%% main skeleton

% preprocess sample matrices

X1 = preprocess_samples(X1, vmean1, w);
X2 = preprocess_samples(X2, vmean2, w);

% construct problem
S = X1 * X2';

switch opts.spectype
    case 'normal'
        if isempty(w)
            tw = n;
        else
            tw = sum(w);
        end
    case 'ratio'
        tw = trace(S * S');
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid spectrum type: %s', opts.spectype);
end


if d > dmax / 3;
    [P1, D, P2] = svd(S, 'econ');
    spectrum = diag(D);
    spectrum = spectrum(1:d);
    P1 = P1(:, 1:d);
    P2 = P2(:, 1:d);
else
    [P1, D, P2] = svds(S, d);
    spectrum = diag(D);
end

% post-process spectrum
if nargout >= 3
    spectrum = spectrum .* spectrum / tw;
end


%% Auxiliary functions

function Xp = preprocess_samples(X, vmean, w)

if ~isequal(vmean, 0)
    if isempty(vmean)
        vmean = slmean(X, w, true);
    end
    Xp = sladdvec(X, -vmean, 1);
else
    Xp = X;
end

if ~isempty(w) 
    Xp = slmulvec(Xp, w, 2);
end


    
    
    









 
 
 