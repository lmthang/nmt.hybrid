function S = edl_readexpdefs(deffile)
%EDL_READEXPDEFS Reads in experiment definition XML file
%
% $ Syntax $
%   - S = edl_readexpdefs(deffile)
%
% $ Arguments $
%   - deffile:      the experiment definition XML file
%   - S:            the struct of the set of experiment definitions
%                   - name:    the name of the definition set
%                   - envconf: the struct array of environment variables
%                   - actions: the list of names of entries
%                   - entry:   the struct array of entries
%                       - name:  the name of action
%                       - type:  the type of action
%                         (scripting | experiment | reporting )
%                       - func:  the matlab function to do the action
%                       - params: the parameters
%
% $ Description $
%   - S = edl_readexpdefs(deffile) reads in a set of experiment definitions
%     from an XML property table file. 
%     The document element should have following attribues:
%       - 'name':       the name of the definition set
%       - 'envconf':    the filename of environment configuration
%     The entries should have following attributes
%       - 'name':   the name of action (experiment)
%       - 'type':   the type of action, the value can be
%                   - 'scripting': generating a prop set controlling experiments
%                   - 'experiment': performing experiments according to a
%                                   property set
%                   - 'reporting': generating a prop set as report 
%       - 'func':   the matlab function to do the action
%       - other attributes as parameters
%   
% $ Remarks $
%   - For experiment action, the params contains following fields:
%       - 'expsch':  the property XML file of the experiment schemes
%   - For scripting and reporting actions, the params contains user-defined
%     fields.
%     
% $ History $
%   - Created by Dahua Lin, on Aug 10th, 2006
%

%% Read file

xdoc = xmlread(deffile);
docelem = xdoc.getDocumentElement;
curdir = fileparts(deffile);

%% Read header

S = [];
S.name = char(docelem.getAttribute('name'));

envconf_fn = char(docelem.getAttribute('envconf'));
if isempty(envconf_fn)
    S.envconf = [];
else
    envconf_fn = sladdpath(envconf_fn, curdir);
    S.envconf = edl_readenvvars(envconf_fn);
end

%% Read Entries

S.entry = [];
S.actions = {};

entries = docelem.getElementsByTagName('Entry');
n = entries.getLength;

if n == 0
    return;
end

S.actions = cell(n, 1);
for i = 1 : n
    
    curentry = entries.item(i-1);
    
    S.entry(i).name = char(curentry.getAttribute('name'));
    S.entry(i).type = char(curentry.getAttribute('type'));
    if ~ismember(S.entry(i).type, {'scripting', 'experiment', 'reporting'})
        error('sltoolbox:parseerror', ...
            'Invalid experiment definition type %s', S.entry(i).type);
    end
    S.entry(i).func = char(curentry.getAttribute('func'));
    S.entry(i).params = xml_getattribs(curentry, {'name', 'type', 'func'});
    
    S.actions{i} = S.entry(i).name;
    
end









