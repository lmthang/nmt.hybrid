function edl_updatectrlfile(guid, filename, idx, status)
%EDL_UPDATECTRLFILE Updates the status in a control file
%
% $ Syntax $
%   - edl_updatectrlfile(guid, filename, idx, status)
%
% $ Arguments $
%   - guid:         the expecting GUID of the control file
%   - filename:     the filename of the control file
%   - idx:          the internal index of the entry to be updated
%   - status:       the updated status
%
% $ Description $
%   - edl_updatectrlfile(guid, filename, idx, status) updates a specified
%     entry of a control file.
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%

%% parse and verify input
if nargin < 4
    raise_lackinput('edl_updatectrlfile', 4);
end

%% read and verify

C = edl_readctrlfile(filename);

if ~strcmpi(C.guid, guid)
    error('edl:interperror', ...
        'Inconsistent between the GUID of control file and script on %s', filename);
end

%% update

n = length(C.status);
if idx > length(C.status)
    error('edl:interperror', ...
        'The index is beyond the number of entries on %s', filename);
end

if ~ismember(status, {'pending', 'succeed', 'failed'})
    error('edl:interperror', ...
        'Invalid status for control file: %s', status);
end

C.status{idx} = status;

%% write


doctag = 'ExpControl';
attribs.guid = guid;
nodetag = 'Entry';

props = struct(...
    'internal_index', cell(n, 1), ...
    'status', cell(n, 1) ...
    );
for i = 1 : n
    props(i).internal_index = i;
    props(i).status = C.status{i};
end


% backup first
copyfile(filename, [filename, '.bak']);
edl_writeprops(doctag, attribs, nodetag, props, filename);





