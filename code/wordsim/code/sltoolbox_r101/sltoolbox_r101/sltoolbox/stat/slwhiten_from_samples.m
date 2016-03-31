function W = slwhiten_from_samples(X, varargin)
%SLWHITEN_FROM_SAMPLES Compute the whitening matrix from sample matrix
%
% $ Syntax $
%   - W = slwhiten_from_samples(X)
%   - W = slwhiten_from_samples(X, ...)
%
% $ Arguments $
%   - X:            the sample matrix
%   - W:            the computed whitening transform matrix
%
% $ Description $
%   - W = slwhiten_from_samples(X) computes the whiten transform matrix
%     from samples using the automatic-selection scheme.
%
%   - W = slwhiten_from_samples(X, ...) computes the whiten transform
%     matrix from samples according to the user-specified properties.
%     \*   
%     \t   Table 1. The properties of sample-whitening computation \\
%     \h     name     &      description                   \\
%           'scheme'  &  The scheme of computation procedure, 
%                        default = 'auto' \\
%           'evproc'  &  The {method, ...} form of eigenvalue processing. 
%                        default = {'std'} \\
%                        This will be input to the slinvevals function. \\
%           'weights' &  The sample weights.  default = []. \\
%     \*      
%     The available schemes are listed as follows
%     \* 
%     \t   Table 2. The schemes of computing whitening matrix       \\
%     \h     name   &     description                               \\
%           'auto'  &  Automatically select a proper scheme for computing.
%                      \\ 
%           'std'   &  Standard scheme: first compute the covariance matrix
%                      and then derive the whitening transform matrix. \\
%           'svd'   &  SVD-based scheme. Using svd for eigen-decomposition.
%           'trans' &  Use a transpose-based way. It is more efficient for
%                      the case with high-dimensionality and small sample
%                      size.                                            \\
%     \*
%     The methods for processing the eigenvalues, i.e. computing their
%     reciprocals can be referred to the slinvevals function.
%
% $ Remarks $
%   - It is a prerequisite that the samples are properly centered. 
%
% $ History $
%   - Created by Lin Dahua on Apr 30, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - replace slmul by slmulvec to increase efficiency
%

%% parse and verify input arguments

if ndims(X) ~= 2
    error('sltoolbox:invaliddims', ...
        'The sample matrix X should be a 2D matrix');
end
n = size(X, 2);

% check options

opts.scheme = 'auto';
opts.evproc = {'std'};
opts.weights = [];
opts = slparseprops(opts, varargin{:});

switch opts.scheme 
    case 'auto'
        fh_compW = @compute_whiten_auto;
    case 'std'
        fh_compW = @compute_whiten_std;
    case 'svd'
        fh_compW = @compute_whiten_svd;
    case 'trans'
        fh_compW = @compute_whiten_trans;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid whiten matrix computing scheme %s', opts.scheme);
end

if ~isempty(opts.weights)
    if ~isequal(size(opts.weights), [1, n])
        error('sltoolbox:sizmismatch', ...
            'The weights should be a 1 x n row vector');
    end
end


%% prepare the samples

if ~isempty(opts.weights)
    X = slmulvec(X, sqrt(max(opts.weights, 0)), 2);
end

%% compute 

W = fh_compW(X, opts.evproc);

%% The functions for computing whiten matrix

function W = compute_whiten_auto(X, evproc)

if size(X, 1) > size(X, 2)
    W = compute_whiten_trans(X, evproc);
else
    W = compute_whiten_std(X, evproc);
end

function W = compute_whiten_std(X, evproc)

S = X * X';
[evs, U] = slsymeig(S);
[revs, U] = proc_eigs(evs, U, evproc);
W = slmulvec(U, sqrt(revs)', 2);

function W = compute_whiten_svd(X, evproc)

[U, D] = svd(X, 0);
evs = diag(D) .^ 2;
clear D;

[revs, U] = proc_eigs(evs, U, evproc);
W = slmulvec(U, sqrt(revs)', 2);

function W = compute_whiten_trans(X, evproc)

S = X' * X;
[evs, V] = slsymeig(S);
U = X * V; 
clear V;
[revs, U] = proc_eigs(evs, U, evproc);
U = slnormalize(U);

W = slmulvec(U, sqrt(revs)', 2);



%% The auxiliary function for obtaining truncated eigen-reciprocals

function [revs, U] = proc_eigs(evs, U, evproc)

revs = slinvevals(evs, evproc{:});

si = find(revs == 0);
if ~isempty(si)
    revs(si) = [];
    U(:, si) = [];
end

