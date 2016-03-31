function D = slkernels2dists(K, ty)
%SLKERNELS2DISTS Computes Euclidean distances from inner products
%
% $ Syntax $
%   - D = slkernels2dists(K)
%   - D = slkernels2dists(K, 'sqr')
%
% $ Arguments $
%   - K:        The inner product matrix (kernel matrix)
%   - D:        The pairwise distance matrix
%
% $ Description $
%   - D = slkernels2dists(K) computes the pairwise distances between 
%     samples based the pairwise inner products between them. The samples
%     are assumed to be in an Euclidean linear space.
%
%   - D = slkernels2dists(K, 'sqr') outputs the square of the distances.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%

%% parse and verify input

if isempty(K) 
    D = [];
    return;
end

if ~isnumeric(K) || ndims(K) ~= 2 || size(K,1) ~= size(K,2)
    error('sltoolbox:invalidarg', ...
        'K should be a square matrix');
end

if nargin >= 2 && strcmpi(ty, 'sqr')
    is_sqr = true;
else
    is_sqr = false;
end

%% compute

D = -(K + K');     

n = size(D, 1);
s = diag(K);

for i = 1 : n
    D(i,:) = D(i,:) + s(i);
end
for i = 1 : n
    D(:,i) = D(:,i) + s(i);
end

D(D < 0) = 0;   % enforce non-negative

if ~is_sqr
    D = sqrt(D);
end
    





    
