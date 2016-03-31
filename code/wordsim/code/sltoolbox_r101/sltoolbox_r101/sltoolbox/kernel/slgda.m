function A = slgda(K, nums, sol)
%SLGDA Performs Baudat's Generalized Discriminant Analysis
%
% $ Syntax $
%   - A = slgda(K, nums)
%   - A = slgda(K, nums, sol)
%
% $ Arguments $
%   - K:        the kernel gram matrix
%   - nums:     the numbers of samples in each classes
%   - sol:      the cell containing the parameter for generalized eigen
%               decomposition
%   - A:        the resulting projection coefficient matrix
%
% $ Description $
%   - A = slgda(K, nums) performs Generalized Discriminant Analysis(GDA),
%     an representative work in using kernel method to extend LDA, 
%     proposed by Baudat et al. The generalized eigen-problem is
%     solved in a default way by slsymgeig.
%
%   - A = slgda(K, nums, sol) in the function, slsymgeig will be invoked 
%     to solve the generalized eigen-decomposition problem. sol is 
%     a cell containing the parameters for slsymgeig. 
%
% $ Remarks $
%   - The function follows the instructions given in the original paper
%     on GDA.
%
%   - The projection is learned after the kernel gram matrix is
%     centralized.
%
%   - The aim of the function is to give an exact implementation of 
%     of representative work GDA, so it does not offer other facilities
%     such as weighting and other ways of scatter computation. For higher
%     flexibility, please use the function slkfd.
%
% $ History $ 
%   - Created by Dahua Lin on May 3rd, 2005
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slkernelscatter', 2);
end

if ndims(K) ~= 2 || size(K, 1) ~= size(K, 2)
    error('sltoolbox:invaliddims', ...
        'The gram matrix K should be a square matrix');
end

M = size(K, 1);         % number of samples
N = length(nums);       % number of classes
if ~isequal(size(nums), [1, N])
    error('sltoolbox:invaliddims', ...
        'The nums should be a 1 x N row vector');
end
if sum(nums) ~= M
    error('sltoolbox:sizmismatch', ...
        'The total number in nums is inconsistent with that in K');
end

if nargin < 3
    sol = {};
end

%% Compute

%% Centralize

K = slcenkernel(K);

%% Construct the eigen-problem

[sp, ep] = slnums2bounds(nums);
W = zeros(M, M);
for i = 1 : N
    ni = nums(i);
    spi = sp(i); epi = ep(i);
    W(spi:epi, spi:epi) = 1 / ni;    
end
clear sp ep;

B = K * W * K;
clear W;

V = K * K;

%% Resolve the eigen-problem

[evs, A] = slsymgeig(B, V, sol{:});
d = sldim_by_eigval(evs);
A = A(:, 1:d);




