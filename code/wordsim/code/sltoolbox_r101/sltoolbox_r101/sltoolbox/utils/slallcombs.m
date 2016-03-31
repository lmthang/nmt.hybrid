function A = slallcombs(nums)
%SLALLCOMBS Generate all combination of numbers
%
% $ Syntax $
%   - A = slallcombs(nums)
%
% $ Arguments $
%   - nums:         the numbers of component sets
%   - A:            the resultant indices matrix
%
% $ Description $
%   - A = slallcombs(nums) generates the set of all possible index-vectors
%     for a d1 x d2 x ... x dK array, where d1, d2, ... dK are stored in 
%     the K-dim vector nums. Then the resultant matrix A would be of size
%     K x dK x ... d2 x d1.
%
% $ Remarks $
%   # If the number in some dimensions equals zero, then an empty array
%     will be returned.
%
% $ Examples $
%   - Generate all indices for 3D array,
%     \{
%           A = slallcombs([3 2 4]);
%     \}
%     Then A is a 3 x 3 x 2 x 4 matrix, with 
%     A(:, i3, i2, i1) = [i1 i2 i3]'
%
% $ History $
%   - Created by Dahua Lin on Nov 19th, 2005
%

%% parse input arguments
nums = nums(:);
K = length(nums);
n = prod(nums);
if n == 0
    A = [];
    return;
end

%% prepare storage
A = zeros(K, n);

%% get organization tables
cprods = cumprod(nums);
n_grps = [1; cprods(1:end-1)];
n_ins = cprods(end) ./ cprods;

%% organize
for i = 1 : K
    
    r = n_ins(i);
    s = nums(i);
    g = n_grps(i);
    
    P = (1:s);
    P = P(ones(r, 1), :);
    P = P(:);
    P = P(:, ones(1, g));
    
    A(i, :) = P(:)';
end

A = reshape(A, [K, nums(end:-1:1)']);

