function [nums, U] = slcount(A)
%SLCOUNT Count the number of sum entities
%
% $ Syntax $
%   - nums = slcount(A)
%   - [nums, U] = slcount(A)
%
% $ Arguments $
%   - A:          the array (or cell array) containing things to be count
%   - nums:       the resultant column vector of numbers
%   - U:          the column vector or column cell array containing
%                 the unique instances
%
% $ Description $
%   - nums = slcount(A) counts the numbers of each entity contained in A.
%
%   - [nums, U] = slcount(A) counts the numbers of each entity, and return
%     the unique instances of these entities in order via U. 
%
% $ Remarks $
%   # The instances belonging to the same type should be put together.
%   # A can be a numeric array or a cell array. For numeric arrays, we
%     use = for comparison, for cell array, we use isequal for comparison.
%
% $ Examples $
%   - Count elements in a numeric array,
%     \{
%          A = [1 1 1 2 2 2 3 3 1 1 1 1];
%          [n, U] = slcount(A)
%
%          n = 
%               3
%               3
%               2
%               4
%
%          U =
%               1
%               2
%               3
%               1
%     \}
%
%   - Count elements in a cell array
%     \{
%          A = {'a', 'a', 'a', 'b', 'b'};
%          [n, U] = slcount(A)
%
%          n =
%              3
%              2
%
%          U =
%              'a'
%              'b'
%     \}
%
% $ History $
%   - Created by Dahua Lin on Nov 19th, 2005
%

if nargout >= 2
    bU = true;
else
    bU = false;
end

if isnumeric(A) % processing numeric array (vectorized code)
    A = A(:);
    difs = diff(A);
    n = length(A);
    nums = [0; find(difs ~= 0); n];
    if bU
        U = A(nums(2:end));
    end
    nums = diff(nums);
    
elseif iscell(A) % processing cell array (non-vectorizable code)
    A = A(:);
    n = length(A);
    difs = false(n, 1);
    for i = 1 : n-1
        if ~isequal(A{i}, A{i+1})
            difs(i) = 1;
        end
    end
    difs(n) = 1;
    nums = [0; find(difs)];
    if bU
        U = A(nums(2:end));
    end
    nums = diff(nums);
    
else
    error('sltoolbox:invalidtype', ...
        'A should be a numeric array or a cell array');
end
    
    










