function [spos, epos] = slnums2bounds(nums)
%SLNUMS2BOUNDS Compute the index-boundaries from section sizes
%
% $ Syntax $
%   - [spos, epos] = slnums2bounds(nums)
%   - spos = slnums2bounds(nums)
%
% $ Description $
%   - [spos, epos] = slnums2bounds(nums) obtains the start positions and
%     end positions of the sections given the number of elements in 
%     the sections.
%
%   - spos = slnums2bounds(nums) only retrieves the start positions.
%
% $ Reamrks $
%   # nums can be either a row vector or a column vector. then the results
%     will be row vector or column vector correspondingly.
%
% $ Examples $
%   - Get boundaries for a 3-section-array
%     \{
%           [s, e] = slnums2bounds([3 2 4])
%
%           s = 
%                1     4     6
%           e =
%                3     5     9
%     \}
%
% $ History $
%   - Created by Dahua Lin on Nov 20th, 2005
%

%% parse and verify input arguments
[d1, d2] = size(nums);
if d1 == 1          % row vector
    iscol = false;
elseif d2 == 1      % column vector
    iscol = true;
else
    error('sltoolbox:notvector', 'nums should be a row/column vector');
end
    

%% compute
cs = cumsum(nums);
if iscol
    spos = [1; cs(1:end-1)+1];
else
    spos = [1, cs(1:end-1)+1];
end
if nargout >= 2
    epos = cs;
end

    

