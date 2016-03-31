function edl_initctrlfile(filename, guidstr, n)
%EDL_INITCTRLFILE Creates an initial control file
%
% $ Syntax $
%   - edl_initctrlfile(filename, guidstr, n)
%
% $ Arguments $
%   - filename:     the destination control filename
%   - guidstr:      the GUID identifying the corresponding script
%   - n:            the number of items
%
% $ Description $
%   - edl_initctrlfile(filename, guidstr, n) writes an initial control
%     file with all status set to pending.
%
% $ History $
%   - Created by Dahua Lin, on Aug 14th, 2006
%

%% parse and verify input

if nargin < 3
    raise_lackinput('edl_initctrlfile', 3);
end

%% Write

doctag = 'ExpControl';
attribs.guid = guidstr;
nodetag = 'Entry';

props = struct(...
    'internal_index', mat2cell((1:n)', ones(n,1)), ...
    'status', repmat({'pending'}, [n, 1]) ...
    );

edl_writeprops(doctag, attribs, nodetag, props, filename)