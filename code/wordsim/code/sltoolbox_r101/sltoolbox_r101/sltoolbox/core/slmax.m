function S = slmax(A, d)
%SLMAX Compute the maximum of values in subarrays
%
% $ Syntax $
%   - S = slmax(A) 
%   - S = slmax(A, d)
%   - S = slmax(A, [d1 d2 ... dk])
%
% $ Arguments $
%   - A:        the input array
%   - d:        the dimensions along which the maximum is searched
%   - S:        the resultant max matrix
%
% $ Description $
%   - S = slmax(A) finds the maximums along column vectors of A. It is 
%     equivalent to S = max(A)
%
%   - S = slmax(A, d) finds the maximums along dimension d. It is 
%     equivalent to S = max(A, [], d)
%
%   - S = slmax(A, [d1 d2 ... dk]) finds the maximum values along dimension
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
    S = max(A, [], d);
else
    k = length(d);
    S = A;
    for i = 1 : k
        S = max(S, [], d(i));
    end
end

        