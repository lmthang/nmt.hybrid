function b = slisguid(v)
%SLISGUID Judges whether the input can represent a GUID
%
% $ Syntax $
%   - b = slisguid(v)
%
% $ Description $
%   - b = slisguid(v) judges whether the input v conforms to the 
%     matlab representation of GUID (1 x 16 uint8 array). 
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

b = isa(v, 'uint8') && isequal(size(v), [1 16]);
