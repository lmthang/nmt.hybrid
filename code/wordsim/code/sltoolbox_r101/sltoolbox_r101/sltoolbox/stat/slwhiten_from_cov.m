function W = slwhiten_from_cov(C, method, varargin)
%SLWHITEN_FROM_COV Compute the whitening transform from covariance matrix
%
% $ Syntax $
%   - W = slwhiten_from_cov(C)
%   - W = slwhiten_from_cov(C, method, ...)
%
% $ Arguments $
%   - C:        the covariance matrix
%   - method:   the method for computing the whitening transform
%   - W:        the computed whitening transform matrix
%
% $ Description $
%   - W = slwhiten_from_cov(C) computes the whitening matrix from C using 
%     default method ('std').
%   - W = slwhiten_from_cov(C, method, r) computes the whitening matrix 
%     from C using specific method and the extra parameters. Please
%     refer to slinvevals for the available methods and corresponding
%     parameters.
%
% $ Remarks $
%   - C should be a positive semidefinite matrix.
%
% $ History $
%   - Created by Dahua Lin on Apr 30th, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - replace slmul by slmulvec to increase efficiency
%

%% parse and verify input arguments

if ndims(C) ~= 2 || size(C, 1) ~= size(C, 2)
    error('sltoolbox:invaliddims', 'C should be a square matrix');
end

if nargin < 2
    method = 'std';
    params = {};
else
    params = varargin;
end


%% compute

[evs, U] = slsymeig(C);
revs = slinvevals(evs, method, params{:})';

if strcmp(method, 'std') 
    si = find(revs > 0);
    revs = revs(si);
    U = U(:, si);
end

W = slmulvec(U, sqrt(revs), 2);


