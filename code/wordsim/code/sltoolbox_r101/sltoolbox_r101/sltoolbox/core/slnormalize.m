function An = slnormalize(A, p, d)
%SLNORMALIZE Normalize the sub-arrays
%
% $ Syntax $
%   - An = slnormalize(A)
%   - An = slnormalize(A, p)
%   - An = slnormalize(A, [], d)
%   - An = slnormalize(A, p, d)
%
% $ Arguments $
%   - A:        the input array
%   - p:        the order of the norm (default = 2)
%   - d:        the dimension of subarrays (default = 1, column vectors)
%   - An:       the normalized array
%
% $ Description $
%   - An = slnormalize(A) normalizes the column vectors of A by 2nd-order.
%
%   - An = slnormalize(A, p) normalizes the column vectors of a by
%     pth-order.
%
%   - An = slnormalize(A, [], d) normalizes the subarrays of A along 
%     dimensions specified by d by 2nd-order.
%
%   - An = slnormalize(A, p, d) normalizes the subarrays of A along
%     dimensions specified by d by pth-order.
%
% $ Remarks $
%   # Normalize an array by pth order means dividing the elements of 
%     the array by its p-th norm.
%   # p can be inf or -inf. If p = inf, then the Lp norm is simply the
%     maximum magnitude value; while if p = -inf, then the Lp norm is the 
%     minimum magnitude value.
%
% $ History $
%   - Created by Dahua Lin on Nov 19th, 2005
%   - Modified by Dahua Lin on Sep 10th, 2006
%       - replace slmul by slmulvec to increase efficiency
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
nrms = slnorm(A, p, d);
nrms = slenforce(nrms, 'positive');
An = slmulvec(A, 1 ./ nrms);



