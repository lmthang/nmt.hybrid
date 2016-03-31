function str = slguidstr(gid)
%SLGUIDSTR Converts a guid to a string 
%
% $ Syntax $
%   - str = slguidstr(gid)
%
% $ Description $
%   - str = slguidstr(gid) converts a GUID represented by a 1 x 16 uint8
%     array to a string in Win32 Registry Format.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

if ~slisguid(gid) 
    error('sltoolbox:invalidarg', ...
        'The input variable should be a GUID');
end

numpat = '%02X';

str = sprintf('%s-%s-%s-%s-%s', ...
    sprintf(numpat, gid(1:4)), ...
    sprintf(numpat, gid(5:6)), ...
    sprintf(numpat, gid(7:8)), ...
    sprintf(numpat, gid(9:10)), ...
    sprintf(numpat, gid(11:16)) ...
    );



