function tf = slisfields(S, fns)
%SLISFIELDS Judges whether the specified fieldnames are fields of S
%
% $ Syntax $
%   - tf = slisfields(S, fns)
%
% $ Arguments $
%   - S:        the struct
%   - fns:      the field names
%   - tf:       the boolean array
%
% $ Description $
%   - tf = slisfields(S, fns) judges whether the names in fns are
%     fieldnames of S. If fns is a char array, then tf is a boolean
%     variable indicating whether fns is a field of S, or if fns is
%     a cell array of field names with k cells, then fns is an array
%     with the same size of fns, and each of its element indicating 
%     whether the corresponding field name is the field of S.
%
% $ History $
%   - Created by Dahua Lin, on Aug 27, 2006
%

if ~isstruct(S)
    error('sltoolbox:invalidarg', ...
        'S should be a struct in order to have fields');
end

if ischar(fns)
    tf = isfield(S, fns);
elseif iscell(fns)
    tf = false(size(fns));
    n = numel(fns);
    for i = 1 : n
        tf(i) = isfield(S, fns{i});
    end
else
    error('sltoolbox:invalidarg', ...
        'fns should be either a char string or a cell array');
end


    