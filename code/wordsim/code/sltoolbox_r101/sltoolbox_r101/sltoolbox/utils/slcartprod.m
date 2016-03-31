function P = slcartprod(varargin)
%SLCARTPROD Get the Cartesian product of a series of sets
%
% $ Syntax $
%   - P = slcartprod(S1, S2, ...)
%
% $ Arguments $
%   - S1, S2, ...:      the component sets in form of cell arrays
%   - P:                the cartesian product of these sets
%
% $ Description $
%   - P = slcartprod(S1, S2, ...) computes the cartesian product of the 
%     sets S1, S2, .... These sets should be in form of cell arrays. If
%     S1, S2, ..., SK respectively have n1, n2, ..., nK elements. Then P 
%     will be an K x nK x ... x n2 x n1 cell array.
%
% $ Reamrks $
%   - If the number of sets is zero, or any set is empty, then P = {} 
%     will be returned.
%
% $ Examples $
%   - Produce the cartesian product of three sets
%     \{
%          S1 = {'a', 'b', 'c'};
%          S2 = {1, 2};
%          S3 = {'A', 'B', 'C', 'D'}
%          P = slcartprod(S1, S2, S3);
%     \}
%     Then P would be a 3 x 4 x 2 x 3 cell array, and P(:, i3, i2, i1)
%     would be a cell array as {S1{i1}, S2{i2}, S3{i3}}. 
%
% $ History $
%   - Created by Dahua Lin on Nov 20th, 2005
%

%% parse and verify input arguments
K = length(varargin);
if K == 0
    P = {};
    return;
end
S = cell(K, 1);
nums = zeros(K, 1);
for i = 1 : K
    if isempty(varargin{i})
        P = {};
        return;
    end
    curset = varargin{i};
    S{i} = curset(:)';          % make it as a row cell array
    nums(i) = length(S{i});
end

%% generate indices
Inds = slallcombs(nums);
Inds = Inds(:, 1:end);

%% get product set
P = cell([K, nums(end:-1:1)']);
for i = 1 : K
    P(i, :) = S{i}(Inds(i, 1:end)); 
end
    

    
    
    
    