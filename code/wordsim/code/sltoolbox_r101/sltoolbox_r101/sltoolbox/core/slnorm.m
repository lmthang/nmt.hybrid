function An = slnorm(A, p, d)
%SLNORM Compute the Lp-norms
%
% $ Syntax $
%   - An = slnorm(A)
%   - An = slnorm(A, p)
%   - An = slnorm(A, [], d)
%   - An = slnorm(A, p, d)
%
% $ Arguments $
%   - A:        the input array
%   - p:        the order of the norm (default = 2)
%   - d:        the dimension of subarrays (default = 1, column vectors)
%   - An:       the resultant array
%
% $ Description $
%   - An = slnorm(A) computes the L2 norm of column vectors of A. If A is
%     an array of size d1 x d2 x ... dn, then An would be of size 
%     1 x d2 x ... dn. Each value of An is the L2 norm of corresponding
%     column vector.
%
%   - An = slnorm(A, p) computes the Lp norm of column vectors of A. If A 
%     is an array of size d1 x d2 x ... dn, then An would be of size 
%     1 x d2 x ... dn. Each value of An is the Lp norm of corresponding
%     column vector.
%
%   - An = slnorm(A, [], d) computes the L2 norm of sub-arrays along the 
%     dimensions specified by d. For example, if A is a d1 x d2 x ... dn
%     array, and d = [1, 2], then An would be an array of size
%     1 x 1 x d3 x ... x dn. Each value of An is the square root of square-
%     sum of values in corresponding plane.
%
%   - An = slnorm(A, p, d) computes the Lp norm of sub-arrays along the 
%     dimensions specified by d.
%
% $ Remarks $
%   # Lp norm is the pth order-root of the sum of p-th power of all values
%     in a subspace. 
%   # p can be inf or -inf. If p = inf, then the Lp norm is simply the
%     maximum magnitude value; while if p = -inf, then the Lp norm is the 
%     minimum magnitude value.
%
% $ History $
%   - Created by Dahua Lin on Nov 19th, 2005
%

%% parse and verify input arguments
if nargin < 2 || isempty(p)
    p = 2;
end
if nargin < 3 || isempty(d)
    d = 1;
end
if ~isscalar(p)
    error('sltoolbox:notscalar', 'p should be a scalar');
end
if p == 0
    error('sltoolbox:zeroerr', 'p should not be zero');
end


%% compute
An = abs(A);
if p == 1
    An = slsum(An, d);
elseif p == 2
    An = sqrt(slsum(An .* An ,d));
elseif p == inf
    An = slmax(An, d);
elseif p == (-inf)
    An = slmin(An, d);
else
    An = slsum(An.^p, d) .^ (1/p);
end


    


    
    






