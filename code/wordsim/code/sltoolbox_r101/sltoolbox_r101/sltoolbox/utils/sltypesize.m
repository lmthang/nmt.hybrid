function s = sltypesize(typename)
%SLTYPESIZE Gets the element size of a specified type
%
% $ Syntax $
%   - s = sltypesize(typename)
%
% $ Arguments $
%   - typename:         the name of a type
%   - s:                the number of bytes in each element of the specified type
%
% $ Description $
%
%   - s = sltypesize(typename) returns the number of bytes of each element
%     in the type specified by typename.
%
%   - The typenames supported are listed below:
%       - 'double'      
%       - 'single'
%       - 'float'
%       - 'uint8'
%       - 'uint16'
%       - 'uint32'
%       - 'uint64'
%       - 'int8'
%       - 'int16'
%       - 'int32'
%       - 'int64'
%       - 'logical'
%       - 'char'
%
% $ History $
%   - Created by Dahua Lin on Dec 7th, 2005
%

switch typename
    case 'double'
        s = 8;
    case {'single', 'float'}
        s = 4;
    case {'uint8', 'int8'}
        s = 1;
    case {'uint16', 'int16'}
        s = 2;
    case {'uint32', 'int32'}
        s = 4;
    case {'uint64', 'int64'}
        s = 8;
    case {'char', 'logical'}
        s = 1;
    otherwise
        error('sltoolbox:invalid_type', ...
            'Unsupported typename %s', typename);
end

