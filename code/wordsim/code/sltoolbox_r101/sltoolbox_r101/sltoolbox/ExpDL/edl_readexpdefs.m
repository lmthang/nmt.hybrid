function ED = edl_readexpdefs(filename)
%EDL_READEXPDEFS Reads in an experiment definition from XML file
%
% $ Syntax $
%   - ED = edl_readexpdefs(filename)
%
% $ Arguments $
%   - filename:     the filename of the experiment definition XML
%   - ED:           the read experiment definition struct
%
% $ Description $
%   - ED = edl_readexpdefs(filename) reads in an experiment definition
%     struct from an XML file. The format of XML can be referred to 
%     the edl.spec.txt. The struct ED has following fields:
%
%       - name: the experiment definition name
%       - selfpath: the absolute path of the experiment definition self
%       - envconf: the filename of environment configuration
%       - env:     the environment configuration struct
%       - scriptdir: the script directory
%       - reportdir: the report directory
%       - mfiledir:  the m-files directory
%       - logfile:   the path of log file
%       - logger:    the logger
%
%       - variables: a struct of all variables, 
%           using variable names as field names
%           using variable values as field values
%       - scripts: the struct of scripts
%           - using name as fieldnames, each field is a struct with
%               - func:     the scripting function
%               - path:     the path of the script file
%               - ctrlpath: the path of the control file
%               - params:   the other parameters
%               - refs:     the struct array of referenced scripts
%                   using role as field name
%                   using script name as field value
%               - refreps:  the struct array of referenced reports
%                   using role as field name
%                   using report name as field value            
%               - refpaths: the struct array of referenced paths
%                   using role as field name
%                   using xml path of (script or report) as field value
%               - refscopes: the struct array of scope names of references
%                            (scripts | reports)
%                   using role as field name
%                   using scope names as field values
%       - experiments: the struct of experiments
%           - using name as fieldnames, each field is a struct with
%               - func:         the experiment function
%               - script:       the script name
%               - scriptpath:   the script filepath
%               - ctrlpath:     the control filepath
%       - reports: the struct array of reports
%           similar to scripts, except for that there is no ctrlpath
%       
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

%% Read file

xdoc = xmlread(filename);
edparent = slfilepart(filename, 'parent');
docelem = xdoc.getDocumentElement;


%% Read Header

headerfns = { ...
    'name', ...
    'envconf', ...
    'scriptdir', ...
    'reportdir', ...
    'mfiledir', ...
    'logfile'};
ED = cell2struct(cell(1, length(headerfns)), headerfns, 2);
headers = xml_getattribs(docelem);
ED = slparseprops(ED, headers);

if slisabspath(filename)
    ED.selfpath = filename;
else
    ED.selfpath = sladdpath(filename, cd());
end

if isempty(ED.envconf)
    ED.env = [];
else
    envpath = sladdpath(ED.envconf, edparent);
    ED.env = edl_readenvvars(envpath);
end

if isempty(ED.env) || ~isfield(ED.env, 'envname')
    error('edl:parseerror', ...
        'The environment must has variable named %s', 'envname');
end
envname = ED.env.envname;

ED.logger = sllog('rootpath', edparent);
if ~isempty(ED.logfile)
    ED.logfile = [ED.logfile, '.', envname, '.log'];
    ED.logger = addfiles(ED.logger, ED.logfile);
end

%% First-Pass Parsing
% 1. Build variable table progressively
% 2. Build basic structures
% 3. Translate variables

ED.variables = [];
ED.scripts = [];
ED.experiments = [];
ED.reports = [];

ED = first_pass(ED, docelem, struct('workdir', ''));


%% Second-Pass Parsing
% 1. Extract paths from cross-references
%   (a) build refpaths for scripts and reports
%   (b) build scriptpath and ctrlpath for experiments
%

% 1.(a)
ED.scripts = build_refpaths(ED, ED.scripts);
ED.reports = build_refpaths(ED, ED.reports);

% 1.(b)
ED.experiments = build_exppaths(ED, ED.experiments);



%% Core function for First-Pass (recursive invoking)

function ED = first_pass(ED, xelem, groupvars)

nodeList = xelem.getChildNodes;
n = nodeList.getLength;

for i = 1 : n   % enumerate all children
    
    node = nodeList.item(i-1);    
    if (node.getNodeType == node.ELEMENT_NODE)       
        tag = char(node.getTagName);
        
        switch tag
            
            case 'Var'
                [varname, varval] = parse_var(ED, node);
                ED.variables.(varname) = varval;
                
            case 'Script'
                [scriptname, scriptstruct] = parse_script(ED, node);
                scriptstruct.params = weak_update(scriptstruct.params, groupvars);
                ED.scripts.(scriptname) = scriptstruct;
            
            case 'Report'
                [reportname, reportstruct] = parse_report(ED, node);
                ED.reports.(reportname) = reportstruct;
                
            case 'Experiment'
                [expname, expstruct] = parse_experiment(ED, node);
                ED.experiments.(expname) = expstruct; 
                
            case 'Group'                
                subgroupvars = parse_group(ED, groupvars, node);
                ED = first_pass(ED, node, subgroupvars);
                
            otherwise
                error('edl:parseerror', ...
                    'Invalid element with tag name %s', tag);                            
        end
                
    end
    
end


%% Core functions for Second-Pass

function S = build_refpaths(ED, S)

fns = fieldnames(S);
n = length(fns);

for i = 1 : n
    fn = fns{i};
    
    S.(fn) = build_refpaths_fortype(ED, fn, S.(fn), 'scripts', 'refs');
    S.(fn) = build_refpaths_fortype(ED, fn, S.(fn), 'reports', 'refreps');
        
end


function S = build_refpaths_fortype(ED, name, S, typeset, typefield)

if ~isfield(S, 'refpaths')
    S.refpaths = [];
end
if ~isfield(S, 'refscopes')
    S.refscopes = [];
end

curset = S.(typefield);
if isempty(curset)
    return;
end

fns = fieldnames(curset);
n = length(fns);
pool = ED.(typeset);

for i = 1 : n
    role = fns{i};    
    refname = curset.(role);
    if ~isfield(pool, refname)
        error('edl:parseerror', ...
            'The reference with role %s (name = %s) in %s is not found in pool %s', ...
            role, refname, name, typeset);
    end
    
    S.refpaths.(role) = pool.(refname).path;    
    S.refscopes.(role) = typeset;
end


function ES = build_exppaths(ED, ES)

if isempty(ES)
    return;
end

fns = fieldnames(ES);
n = length(fns);

for i = 1 : n
    fn = fns{i};
    scriptname = ES.(fn).script;
    if ~isfield(ED.scripts, scriptname)
        error('edl:parseerror', ...
            'The referred script named %s for %s is not found', ...
            scriptname, fn);
    end
    
    curscript = ED.scripts.(scriptname);
    
    ES.(fn).scriptpath = curscript.path;
    ES.(fn).ctrlpath = curscript.ctrlpath;
end



%% Node Parsing functions

% variable parsing

function [vname, vval] = parse_var(ED, xelem)

vname = make_name(ED, 'variables', xelem);
vval = get_tattrib(ED, xelem, 'variables', vname, 'val');


% script parsing

function [sname, scr] = parse_script(ED, xelem)

sname = make_name(ED, 'scripts', xelem);

scr.func = get_tattrib(ED, xelem, 'scripts', sname, 'func');
scr.func = sladdpath(scr.func, ED.mfiledir);

scr.path = '';
scr.ctrlpath = '';
scr = update_fields(ED, scr, xelem, {'path', 'ctrlpath'});
scr.params = get_tattribs(ED, xelem, ...
    'exclude', {'name', 'func', 'path', 'ctrlpath'});

if isempty(scr.path)
    scr.path = [sname, '.script.xml'];
end
if isempty(scr.ctrlpath)
    scrfiletitle = slfilepart(scr.path, 'title');
    scr.ctrlpath = slchangefilepart(scr.path, 'title', [scrfiletitle, '.control']);
end

scr.path = sladdpath(scr.path, ED.scriptdir);

scr = build_reftables(ED, scr, xelem);


% report parsing

function [rname, rep] = parse_report(ED, xelem)

rname = make_name(ED, 'reports', xelem);

rep.func = get_tattrib(ED, xelem, 'reports', rname, 'func');
rep.func = sladdpath(rep.func, ED.mfiledir);

rep.path = '';
rep = update_fields(ED, rep, xelem, {'path'});
rep.params = get_tattribs(ED, xelem, ...
    'exclude', {'name', 'func', 'path', 'ctrlpath'});

if isempty(rep.path)
    rep.path = [rname, '.report.xml'];
end
rep.path = sladdpath(rep.path, ED.reportdir);

rep = build_reftables(ED, rep, xelem);


% experiment parsing

function [expname, es] = parse_experiment(ED, xelem)

expname = make_name(ED, 'experiments', xelem);

es.func = get_tattrib(ED, xelem, 'experiments', expname, 'func'); 
es.func = sladdpath(es.func, ED.mfiledir);

es.script = '';
es = update_fields(ED, es, xelem, {'script'});

if isempty(es.script)
    es.script = expname;
end


% group parsing

function newvars = parse_group(ED, gvars, xelem)

newvars = get_tattribs(ED, xelem, 'exclude', {'title'});

if ~isfield(newvars, 'workdir') || isempty(newvars.workdir)
    newvars.workdir = gvars.workdir;
else
    newvars.workdir = sladdpath(newvars.workdir, gvars.workdir);
end

newvars = weak_update(newvars, gvars);






%% Auxiliary functions


function val = translate_attribval(ED, attrval)

val = attrval;
if length(val) > 1 && val(1) == '$'
    varname = val(2:end);
    if ~isfield(ED.variables, varname)
        error('edl:parseerror', ...
            'The variable %s is not found', varname);
    end
    val = ED.variables.(varname);
end


function aval = get_tattrib(ED, xelem, scope, elemname, attrname)

if xelem.hasAttribute(attrname)
    aval = char(xelem.getAttribute(attrname));
    aval = translate_attribval(ED, aval);
else
    error('edl:parseerror', ...
        'The attribute %s is not found in %s of %s', attrname, elemname, scope);
end


function A = get_tattribs(ED, xelem, varargin)

A = xml_getattribs(xelem, varargin{:});

if ~isempty(A)
    fns = fieldnames(A);
    n = length(fns);
    for i = 1 : n
        fn = fns{i};
        A.(fn) = translate_attribval(ED, A.(fn));
    end
end    


function name = make_name(ED, scope, xelem)

if xelem.hasAttribute('name')
    name = char(xelem.getAttribute('name'));
    if isfield(ED.(scope), name)
        error('edl:parseerror', ...
            'Redefinition of %s in %s', name, scope);
    end
else
    error('edl:parseerror', ...
        'Encounter an element without name in %s', scope);
end


function S = update_fields(ED, S, xelem, fns)

S = slparseprops(S, get_tattribs(ED, xelem, ...
    'select', fns, ...
    'forceexist', false));


function S = weak_update(S, Scomp)

fns = fieldnames(Scomp);
n = length(fns);
for i = 1 : n
    fn = fns{i};
    if ~isfield(S, fn)
        S.(fn) = Scomp.(fn);
    end
end


function S = build_reftables(ED, S, xelem)

refNodes = xelem.getElementsByTagName('Ref');
nrs = refNodes.getLength;

S.refs = [];
S.refreps = [];

for i = 1 : nrs
    curnode = refNodes.item(i-1);
    cur = get_tattribs(ED, curnode, 'select', {'role', 'name'});
    S.refs.(cur.role) = cur.name;
end

refrepNodes = xelem.getElementsByTagName('RefReport');
nrr = refrepNodes.getLength;

for i = 1 : nrr
    curnode = refrepNodes.item(i-1);
    cur = get_tattribs(ED, curnode, 'select', {'role', 'name'});
    S.refreps.(cur.role) = cur.name;
end








