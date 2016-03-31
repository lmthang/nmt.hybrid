function M = sltensor_unfold(T, k)
%SLTENSOR_UNFOLD Unfolds a tensor to a matrix 
%
% $ Syntax $
%   - M = sltensor_unfold(T, k)
%
% $ Arguments $
%   - T:        the tensor
%   - k:        the dimension along which the tensor is unfolded
%   - M:        the matrix obtained by unfolding the tensor
%
% $ Description $
%   - M = sltensor_unfold(T, k) Unfolds a tensor T to the matrix M along
%   the k-th dimension.
%
% $ History $
%   - Created by Dahua Lin on Dec 17th, 2005
%

%% parse and verify
if nargin < 2
    raise_lackinput('sltensor_unfold', 2);
end

%% compute
n = ndims(T);
if k == 1
    M = T(1:end, :);
elseif k <= n
    if k < n
        pdims = [k, 1:k-1, k+1:n];
    else
        pdims = [k, 1:k-1];
    end
    M = permute(T, pdims);
    M = M(1:end, :);
else
    M = T(:)';
end






    

