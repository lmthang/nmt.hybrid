function S = slmin(A, d)
%SLMIN Compute the minimum of values in subarrays
%
% $ Syntax $
%   - S = slmin(A) 
%   - S = slmin(A, d)
%   - S = slmin(A, [d1 d2 ... dk])
%
% $ Arguments $
%   - A:        the input array
%   - d:        the dimensions along which the minimum is searched
%   - S:        the resultant min matrix
%
% $ Description $
%   - S = slmin(A) finds the minimums along column vectors of A. It is 
%     equivalent to S = min(A)
%
%   - S = slmin(A, d) finds the minimums along dimension d. It is 
%     equivalent to S = min(A, [], d)
%
%   - S = slmin(A, [d1 d2 ... dk]) finds the minimum values along dimension
%     d1, d2, ... dk.
%
% $ History $
%   - Created by Dahua Lin on Nov 19th, 2005
%

%% parse and verify input arguments
if nargin < 2 || isempty(d)
    d = 1;
end

%% compute
if isscalar(d)
    S = min(A, [], d);
else
    k = length(d);
    S = A;
    for i = 1 : k
        S = min(S, [], d(i));
    end
end

        