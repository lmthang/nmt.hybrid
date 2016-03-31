function C = edl_readctrlfile(filename)
%EDL_READCTRLFILE Reads in a control file 
%
% $ Description $
%   - C = edl_readctrlfile(filename)
%
% $ Arguments $
%   - filename:     the filename of the control file
%   - C:            the struct of the read information
%                   - guid:    the GUID string     
%                   - status:  the n x 1cell array of status
%
% $ Description $
%   - C = edl_readctrlfile(filename) reads in a control file.
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%

%% Read in file

doctag = 'ExpControl';
nodetag = 'Entry';

S = edl_readprops(filename, nodetag);

%% Post-Processing

% doc tag
if ~strcmp(S.tag, doctag)
    error('edl:parseerror', ...
        'Invalid document tag %s for control file', S.tag);
end

% doc attribs
if isempty(S.attribs)
    error('edl:parseerror', ...
        'The document element for control file has no attributes');
end

attrnames = {'guid'};
tf = isfield(S.attribs, attrnames);
if ~all(tf)
    error('edl:parserror', ...
        'The required header %s does not exist', ...
        attrnames{find(~tf, 1)});
end

C.guid = S.attribs.guid;

% entries

edl_check_internalindices(S.(nodetag));
C.status = {S.(nodetag).status};
C.status = C.status(:)';



