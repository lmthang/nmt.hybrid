function A = slexpand(nums, U)
%SLEXPAND Expand a set to multiple instance
%
% $ Syntax $
%   - A = slexpand(nums)
%   - A = slexpand(nums, U)
%
% $ Arguments $
%   - nums:           the numbers of instances for entities
%   - U:              the unique set of entities (numeric vector or cell arr)
%   - A:              the expanded array
%
% $ Description $
%   - A = slexpand(nums) expands the one-based labels by numbers specified
%     in nums.
%
%   - A = slexpand(nums, U) expands the entity-set U by numbers specified
%     in nums.
%
% $ Remarks $
%   # nums can be either column vector or row vector. Then, A would be
%     column array or row array correspondingly.
%   # U can be numeric array or cell array. Then A would be numeric array
%     or cell array correspondingly.
%   # If U is not specified, it is equivalent to set U = [1, 2, ...];
%
% $ Examples $
%   - Expand one-based labels,
%     \{
%          A = slexpand([3 2 4])
%
%          A = 
%               1     1     1     2     2     3     3     3     3
%     \}
%
%   - Expand a designated set
%     \{
%          A = slexpand([2; 3], [10 20])
%
%          A = 
%               10
%               10
%               20
%               20
%               20
%     \}
%
%   - Expand a cell array
%     \{
%         A = slexpand([2 3], {'a', 'b'})
%
%         A = 
%              'a'    'a'    'b'    'b'    'b'
%     \}
%
% $ History $
%   - Created by Dahua Lin on Nov 19th, 2005
%

%% parse and verify input arguments
[d1, d2] = size(nums);
if d1 == 1 % row vector
    c = d2;
    iscol = false;
elseif d2 == 1
    c = d1;
    iscol = true;
else
    error('sltoolbox:notvector', 'nums should be a vector');
end
if nargin < 2 || isempty(U)
    U = 1:c;
end
if numel(U) < c
    error('sltoolbox:notenoughelems', 'U should have at least c elements');
end
n = sum(nums(:));

%% prepare container
if isnumeric(U)
    if iscol
        A = zeros(n, 1);
    else
        A = zeros(1, n);
    end
elseif iscell(U)
    if iscol
        A = cell(n, 1);
    else
        A = cell(1, n);
    end
else
    error('sltoolbox:invalidtype', 'U should be a numeric array or a cell array');
end

%% execute expanding

p2 = 0;
for k = 1 : c
    p1 = p2 + 1;
    p2 = p2 + nums(k);
    
    A(p1:p2) = U(k);    
end



    





