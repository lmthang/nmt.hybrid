function script = edl_readscript(filename)
%EDL_READSCRIPT Reads in a EDL script
%
% $ Syntax $
%   - script = edl_readscript(filename)
%
% $ Description $
%   - script = edl_readscript(filename) reads in a EDL script from 
%     a script xml file. The returned script is a struct with
%     following fields:
%       - attribs: the header attributes
%           - guid:     the GUID string identifying the script
%           - workdir:  the root work diretory of experiments
%           - ctrlpath: the corresponding control file path
%       - entries:  the experiment parameter entries
%         at least have following fields:
%           - internal_index:  the internal index
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%

%% Read in file

doctag = 'ExpScript';
nodetag = 'Entry';

S = edl_readprops(filename, nodetag);

%% Post-Processing

% doc tag
if ~strcmp(S.tag, doctag)
    error('edl:parseerror', ...
        'Invalid document tag %s for script', S.tag);
end

% doc attribs
if isempty(S.attribs)
    error('edl:parseerror', ...
        'The document element for script has no attributes');
end

attrnames = {'guid', 'workdir', 'ctrlpath'};
tf = isfield(S.attribs, attrnames);
if ~all(tf)
    error('edl:parserror', ...
        'The required header %s does not exist', ...
        attrnames{find(~tf, 1)});
end

script.attribs = struct(...
    'guid', S.attribs.guid, ...
    'workdir', S.attribs.workdir, ...
    'ctrlpath', S.attribs.ctrlpath);

% entries

script.entries = edl_check_internalindices(S.(nodetag));






