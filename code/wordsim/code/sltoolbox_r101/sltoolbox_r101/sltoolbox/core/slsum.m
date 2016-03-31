function S = slsum(A, d)
%SLSUM Compute the sum of values in subarrays
%
% $ Syntax $
%   - S = slsum(A) 
%   - S = slsum(A, d)
%   - S = slsum(A, [d1 d2 ... dk])
%
% $ Arguments $
%   - A:        the input array
%   - d:        the dimensions along which the sum is performed
%   - S:        the resultant sum matrix
%
% $ Description $
%   - S = slsum(A) sums up the values in column vectors of A. It is 
%     equivalent to S = sum(A)
%
%   - S = slsum(A, d) sums up the values along dimension d. It is 
%     equivalent to S = sum(A, d)
%
%   - S = slsum(A, [d1 d2 ... dk]) sums up the values along dimension
%     d1, d2, ... dk.
%
% $ History $
%   - Created by Dahua Lin on Nov 18th, 2005
%

%% parse and verify input arguments
if nargin < 2 || isempty(d)
    d = 1;
end

%% compute
if isscalar(d)
    S = sum(A, d);
else
    k = length(d);
    S = A;
    for i = 1 : k
        S = sum(S, d(i));
    end
end

        