function Y = slapplypca(S, X)
%SLAPPLYPCA Applies PCA model to samples
%
% $ Syntax $
%   Y = slapplypca(S, X)
%
% $ Arguments $
%   - S:        the struct representing the PCA model 
%   - X:        the sample matrix
%   - Y:        the principal component vectors of the samples
%
% $ Description $
%   - Y = slapplypca(S, X) applies the PCA model S to reduce the
%     vector dimensions of samples X. It outputs the PCA features by Y.
%     The formula for transform for each sample x in X is:
%     y = S.P' * (x - S.vmean)
%
% $ History $
%   - Created by Dahua Lin on May 1st, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - change sladd to sladdvec
%

%% parse and verify input arguments
if nargin < 2
    raise_lackinput('slapplypca', 2);
end

if ndims(X) ~= 2
    error('sltoolbox:invaliddims', ...
        'The sample matrix X should be a 2D matrix');
end

if size(X, 1) ~= S.sampledim
    error('sltoolbox:sizmismatch', ...
        'The sample dimension does not match that of PCA model');
end

%% compute
Y = S.P' * sladdvec(X, -S.vmean, 1);
