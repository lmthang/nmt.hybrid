function edl_writescript(filename, guid, workdir, ctrlpath, props)
%EDL_WRITESCRIPT Writes an EDL script 
%
% $ Syntax $
%   - edl_writescript(filename, guid, workdir, ctrlpath, props)
%
% $ Arguments $
%   - filename:     the filename of the destination script
%   - guid:         the guid string assigned to the script
%   - workdir:      the root working directory of the experiments
%   - ctrlpath:     the path of control file (r.t. script's parent) 
%   - props:        the properties of the experiment parameters
%                   (no need of internal_index)
%
% $ Description $
%   - edl_writescript(filename, guid, workdir, ctrlpath, props) writes the
%     script according to the information provided. It will also add the
%     internal index to each entry.
%
% $ Remarks $
%   - The function will also creates the initial control file.
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%

%% parse and verify input

if nargin < 5
    raise_lackinput('edl_writescript', 5);
end

%% Prepare elements

doctag = 'ExpScript';
nodetag = 'Entry';

attribs.guid = guid;
attribs.workdir = workdir;
attribs.ctrlpath = ctrlpath;

n = length(props);
for i = 1 : n
    props(i).internal_index = i;
end


%% Write

edl_writeprops(doctag, attribs, nodetag, props, filename);
cpath = sladdpath(ctrlpath, slfilepart(filename, 'parent'));
edl_initctrlfile(cpath, guid, n);

