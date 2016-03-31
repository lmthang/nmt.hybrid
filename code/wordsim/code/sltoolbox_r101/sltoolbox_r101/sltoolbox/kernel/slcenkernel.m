function KC = slcenkernel(K0, K, w)
%SLCENKERNEL Compute the centralized kernel matrix
%
% $ Syntax $
%   - KC = slcenkernel(K0)
%   - KC = slcenkernel(K0, [], w)
%   - KC = slcenkernel(K0, K)
%   - KC = slcenkernel(K0, K, w)
%
% $ Arguments $
%   - K0:       the gram matrix of the referenced samples
%   - K:        the kernel matrix for target samples
%   - w:        the weights for the referenced samples
%   - KC:       the centralized kernel matrix.
%
% $ Description $
%   - KC = slcenkernel(K0) compute the centralized kernel matrix from
%     the original kernel gram matrix K0.
% 
%   - KC = slcenkernel(K, [], w) compute the centralized kernel gram
%     matrix from the original kernel gram matrix K0. The mean feature
%     is obtained with the weights for referenced samples, given by w.
%
%   - KC = slcenkernel(K0, K) compute the centralized kernel matrix
%     for target samples, with the original gram matrix for referenced
%     samples K0 and the kernel matrix for target samples w.r.t the 
%     referenced samples K given.
%
%   - KC = slcenkernel(K0, K, w) compute the centralied kernel matrix
%     for target samples, with feature mean computed in a weighted 
%     manner.
%
% $ Remarks $
%   -# For original kernel matrix K, it is defined as 
%      K(i, j) = <phi(i), phi(j)>, given that phi(i) and phi(j) are
%      the feature map of the samples x(i) and x(j) respectively.
%      Then the centralized kernel matrix is defined as
%      KC(i, j) = <phi(i) - mean_phi, phi(j) - mean_phi>, where
%      mean_phi is the mean of all referenced feature maps. If w
%      is specified mean_phi is given by weighted mean. 
%      Kernel centralization plays an important role in many 
%      kernelized algorithms such as Kernel PCA.
%
%   -# Suppose the mean feature map is defined by 
%      mean_phi = sum_i w_i phi(i).
%      Then the centralized kernel gram matrix can be written as
%      KC = K - 1 * (w^T * K) - (K * w) * 1^T + 1 * (w^T * K * w) * 1^T.
%      It can be easily shown that the mean of columns (rows) of KC is
%      a zero vector. Thus, centralize the centralized kernel matrix
%      would keep the input unchanged. 
%
%   -# Instead of applying the formula given above, the function 
%      implements a more efficient computational routine by reducing
%      the redundant computations.
%
% $ History $
%   - Created by Dahua Lin on May 2nd, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - use sladdrowcols to replace sladd to increase efficiency
%

%% parse and verify input arguments

% for K0
n0 = size(K0, 1);
if ndims(K0) ~= 2 || size(K0, 2) ~= n0;
    error('sltoolbox:invaliddims', ...
        'K0 should be a 2D square matrix');
end

% for K
if nargin < 2 || isempty(K)
    K = K0;
else
    if ndims(K) ~= 2
        error('sltoolbox:invaliddims', ...
            'K should be a 2D matrix');
    end
    if size(K, 1) ~= n0
        error('sltoolbox:sizmismatch', ...
            'Size inconsistency between K0 and K');
    end
end
    
% for w    
if nargin < 3 || isempty(w)
    isweighted = false;
else
    if ~isequal(size(w), [1, n0])
        error('sltoolbox:sizmismatch', ...
            'Size inconsistency between K0 and w');
    end    
    isweighted = true;
end


%% compute

% Steps:
% 1. compute v1: mean row vector of K (1 x n)
% 2. compute v2: mean column vector of K0 (n x 1)
% 3. compute s3: mean value of of all elements of K0 (1 x 1)
% 4. KC = K - expand(v1) - expand(v2) + s3

if ~isweighted  % non-weighted case

    v1 = sum(K, 1) * (1 / n0);
    v2 = sum(K0, 2) * (1 / n0);
    s3 = sum(v2) * (1 / n0);
    
else            % weighted case
    
    w = w / sum(w);     % normalize the weights
    
    v1 = w * K;
    v2 = K0 * w';
    s3 = w * v2;
    
end

KC = sladdrowcols(K, -v1, -v2+s3);


