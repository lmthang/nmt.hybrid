function report = edl_readreport(filename)
%EDL_READREPORT Reads in a EDL report
%
% $ Syntax $
%   - report = edl_readreport(filename)
%
% $ Description $
%   - report = edl_readreport(filename) reads in a EDL report from 
%     a report xml file. The returned report is a struct with
%     following fields:
%       - attribs: the header attributes
%           - guid:     the GUID string identifying the report
%       - entries:  the report item entries
%         at least have following fields:
%           - internal_index:  the internal index
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%

%% Read in file

doctag = 'ExpReport';
nodetag = 'Entry';

S = edl_readprops(filename, nodetag);

%% Post-Processing

% doc tag
if ~strcmp(S.tag, doctag)
    error('edl:parseerror', ...
        'Invalid document tag %s for report', S.tag);
end

% doc attribs
if isempty(S.attribs)
    error('edl:parseerror', ...
        'The document element for report has no attributes');
end

attrnames = {'guid'};
tf = isfield(S.attribs, attrnames);
if ~all(tf)
    error('edl:parserror', ...
        'The required header %s does not exist', ...
        attrnames{find(~tf, 1)});
end

report.attribs = S.attribs;

% entries

report.entries = edl_check_internalindices(S.(nodetag));






