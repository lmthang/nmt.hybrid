function T = sltensor_fold(M, dims, k)
%SLTENSOR_FOLD Folds a matrix into a tensor
%
% $ Syntax $
%   - T = sltensor_fold(M, dims, k)
%
% $ Arguments $
%   - M:                the unfolded matrix of the tensor
%   - dims:             the dimension sizes of the tensor
%   - k:                the dimension along which the tensor is folded
%   - T:                the resultant tensor
%
% $ Description $
%   - T = sltensor_fold(M, dims, k) folds the matrix M to a tensor with
%   its mode dimensions specified in dims along the k-th mode.
%
% $ History $
%   - Created by Dahua Lin on Dec 17th, 2005
%

%% parse and verify
if nargin < 2
    raise_lackinput('sltensor_fold', 3);
end
n = length(dims);
if k < 1
    error('sltoolbox:invalidarg', ...
        'The mode index should be positive');
elseif k > n
    error('sltoolbox:argmismatch', ...
        'The mode index exceeds the tensor order');
end
if numel(M) ~= prod(dims)
    error('sltoolbox:argmismatch', ...
        'The size of matrix and the dimensions of tensor do not match');
end

%% compute
if k == 1
    T = reshape(M, dims);
else
    if k < n
        pdims = [k, 1:k-1, k+1:n];
    else
        pdims = [k, 1:k-1];
    end
    T = reshape(M, dims(pdims));
    T = ipermute(T, pdims);
end

