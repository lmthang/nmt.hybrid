function R = slinvcov(C, method, r)
%SLINVCOV Compute the inverse of an covariance matrix
%
% $ Syntax $
%   - R = slinvcov(C)
%   - R = slinvcov(C, method, r)
%
% $ Arguments $
%   - C:        the covariance matrix (matrices)
%   - method:   the method of inverse calculation
%   - r:        the additional parameter for computation
%   - R:        the computed inverse matrix
%
% $ Description $
%   - R = slinvcov(C) computes the inverse of C using default method. If
%     C is d x d x ... array, then R would be an array of the same size.
%     Each page of R is the inverse matrix of the corresponding covariance
%     matrix in C.
%
%   - R = slinvcov(C, method, r) computes the inverse of C using specific
%     method. For some method, extra parameters are needed, which can be
%     given subsequently. The method can be either 'direct', i.e. directly
%     invoke inv for inverse computing or the names specified in
%     slinvevals. For the latter cases, we adopt the following formula
%     for computation:
%       C^(-1) = U * diag(1 ./ evals) * U^T
%     while the reciprocals of eigenvalues are computed in a robust way
%     by the methods available in slinvevals. The default method is
%     'direct'.
%
% $ Remarks $
%   - C should be a symmetric and positive semidefinite matrix.
%
% $ History $
%   - Created by Dahua Lin on Apr 22, 2006
%   - Modified by Dahua Lin on Apr 30, 2006
%     - Base on the slinvevals function to eigenvalue processing.
%     - Re-organize the code in a clearer way
%

%% parse and verify input arguments

if size(C, 1) ~= size(C, 2)
    error('sltoolbox:invalidarg', 'C should be symmetric matrix (matrices)');
end

% for method
if nargin < 2 || isempty(method)
    method = 'direct';
end

if nargin < 3
    r = [];
end


%% delegate to the computation routine

switch method
    case 'direct'
        R = frmroutine_for_getinv(C, @inv, {});
    case {'pseudo', 'std'}
        R = frmroutine_for_getinv(C, @compinv_evd_based, {'std', r});
    case 'reg'
        R = frmroutine_for_getinv(C, @compinv_evd_based, {'reg', r});
    case 'bound'
        R = frmroutine_for_getinv(C, @compinv_evd_based, {'bound', r});
    case 'gapprox'
        R = frmroutine_for_getinv(C, @compinv_evd_based, {'gapprox', r});
end


%% the framework routine to compute
function R = frmroutine_for_getinv(C, fh, params)
% fh is the function handle for computing single inverse

if ndims(C) == 2  % single matrix    
    M = 0.5 * (C + C');  % enforce symmetry
    R = fh(M, params{:});
else
    siz = size(C);
    n = prod(siz(3:end));
    R = zeros(siz);
    
    for i = 1 : n
        M = C(:,:,i);
        M = 0.5 * (M + M');  % enforce symmetry
        R(:,:,i) = fh(M, params{:});
    end
end


%% the general routine for the inverse computation based on eigen-decompose
function R = compinv_evd_based(C, method, r)

% eigen-decompose
[ev, V] = slsymeig(C);

% compute the reciprocals of the eigenvalues
ev = slinvevals(ev, method, r);
D = diag(ev);

% construct the inverse matrix
R = V * D * V';



