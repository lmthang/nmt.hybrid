function C = slrange2indcells(range)
%SLRANGE2INDCELLS Converts a range array to indices cell array
%
% $ Syntax $
%   - C = slrange2indcells(range)
%
% $ Arguments $
%   - range:        the 2 x d array specifying the range
%   - C:            the 1 x d indices array
%
% $ Description $
%   - C = slrange2indcells(range) converts an range array to a cell array
%     of indices. 
%     For example, for a range array in the form:
%     [s1, s2, ..., sd; e1, e2, ..., ed], it will outputs the cell array as
%     C = {s1:e1, s2:e2, ..., sd:ed}.
%     So that A(C{:}) will give A(s1:e1, s2:e2, ..., sd:ed).
%
% $ History $
%   - Created by Dahua Lin, on Jul 29th, 2006
%

d = size(range, 2);
C = cell(1, d);

for i = 1 : d
    C{i} = range(1, i) : range(2, i);
end

