function edl_writereport(filename, guid, props)
%EDL_WRITEREPORT Writes an EDL report 
%
% $ Syntax $
%   - edl_writereport(filename, guid, props)
%
% $ Arguments $
%   - filename:     the filename of the destination report
%   - guid:         the guid string assigned to the report
%   - props:        the properties of the report items
%                   (no need of internal_index)
%
% $ Description $
%   - edl_writereport(filename, guid, workdir, ctrlpath, props) writes the
%     report according to the information provided. It will also add the
%     internal index to each entry.
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%

%% parse and verify input

if nargin < 3
    raise_lackinput('edl_writereport', 3);
end

%% Prepare elements

doctag = 'ExpReport';
nodetag = 'Entry';

attribs.guid = guid;

n = length(props);
for i = 1 : n
    props(i).internal_index = i;
end


%% Write

edl_writeprops(doctag, attribs, nodetag, props, filename);


