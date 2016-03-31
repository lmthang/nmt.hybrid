function A = slinitarray(type, size)
%SLINITARRAY Initialize an array of specified type and size
%
% $ Syntax $
%   - A = slinitarray(type, size)
%
% $ Arguments $
%   - type          the string representing the element type of the array
%   - size          the vector representing the size of the array
%   - A             the initialized array
%
% $ Description $
%   - A = slinitarray(type, size) creates an array of specified type and
%     given size, with all elements being zeros.
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


switch type
    case 'double'
        A = zeros(size);
    case {'single', 'float'}
        A = zeros(size, 'single');
    case {'uint8', 'uint16', 'uint32', 'uint64', ...
          'int8', 'int16', 'int32', 'int64'}
        A = zeros(size, type);
    case 'logical'
        A = false(size);
    case 'char'
        A = char(zeros(size, 'uint8'));
    otherwise
        error('sltoolbox:invalid_type', ...
            'Unsupported typename %s', typename);
end

