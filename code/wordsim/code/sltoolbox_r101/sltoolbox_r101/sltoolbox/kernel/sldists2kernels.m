function K = sldists2kernels(D, ty)
%SLDISTS2KERNELS Computes the inner products from distances
%
% $ Syntax $
%   - K = sldists2kernels(D)
%   - K = sldists2kernels(D, 'sqr')
%
% $ Arguments $
%   - D:        The pairwise distance matrix
%   - K:        The pairwise inner product matrix (kernel matrix)
%   
% $ Description $
%   - K = sldists2kernels(D) computes the inner products between samples
%     pairwisely with the pairwise norms given. Assume that the samples 
%     are in an Euclidean linear space, and are centralized.
%
%   - K = sldists2kernels(D, 'sqr') performs the calculation with the 
%     input D matrix containing the square distances.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%

%% parse and verify input

if isempty(D) 
    K = [];
    return;
end

if ~isnumeric(D) || ndims(D) ~= 2 || size(D,1) ~= size(D,2)
    error('sltoolbox:invalidarg', ...
        'D should be a square matrix');
end

if nargin >= 2 && strcmpi(ty, 'sqr')
    is_sqr = true;
else
    is_sqr = false;
end

%% compute

% preprocess

if ~is_sqr
    K = D .* D;     % make squares
else
    K = D;
end
K = 0.5 * (K + K'); % enforce symmetry

% compute
n = size(K, 1);

s = sum(K, 1);
t = sum(s);

s = s / n;
t = t / (n*n);

for i = 1 : n
    K(i,:) = K(i,:) - s(i);
end
for i = 1 : n
    K(:,i) = K(:,i) - s(i);
end
K = K + t;

K = (-0.5) * K;
    



    
    
    
    
    

