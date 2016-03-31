function edl_go(expdef, type, name, filter, runopt)
%EDL_GO The Top interface for doing experiments in EDL
%
% $ Syntax $
%   - edl_go(expdef, type, name)
%   - edl_go(expdef, type, name, filter, runopt)
%
% $ Arguments $
%   - expdef:       the experiment definition
%   - type:         the type of the action
%                   'script'|'experiment'|'report'
%   - name:         the name of action to take
%   - filter:       the filter of selecting experiment part
%   - runopt:       the running option
%                   'restart':  run all specified experiments 
%                               (default follows edl_batchexp)
%                   'resume':   run the experiments that not succeeded
%
% $ Description $
%   - edl_go(expdef, type, name) takes a specified action identified 
%     by its type and name. expdef can be a experiment definition 
%     file or a loaded structure.
%
%   - edl_go(expdef, type, name, filter) takes a specified action and 
%     selects the filtered parts to run. Such a syntax is only applicable 
%     for 'experiment' action.
%
% $ Remarks $
%   - This function should be started from experiment root.
%
% $ History $
%   - Created by Dahua Lin, on Aug 10th, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('edl_go', 3);
end

need_close_logger = false;
if ischar(expdef)
    expdef = edl_readexpdefs(expdef);
    need_close_logger = true;
end
if ~isstruct(expdef)
    error('sltoolbox:invalidarg', ...
        'Experiment definition should either be a filename or the loaded struct');
end

if nargin < 4
    filter = [];
end

if nargin < 5
    runopt = [];
end

current_folder = cd();

%% Main Procedure wrapper

logger = expdef.logger;

try
    main_proc_go(expdef, type, name, filter, logger, runopt);
catch
    err = lasterror();
    edl_logerror('edl_go', err, logger);
    lasterror('reset');    
end

if need_close_logger
    close(logger);
end

cd(current_folder);


%% Main Procedure function

function main_proc_go(expdef, type, name, filter, logger, runopt)

rootdir = slfilepart(expdef.selfpath, 'parent');

write(logger, 'on Experiment Definition: %s', expdef.name);
writeblank(logger);

switch type
    case 'script'
        scope = 'scripts';
        curscript = take_element(expdef, scope, name);
        
        % output script info
        
        gidstr = init_instance(curscript, ...
            sprintf('Starting scripting %s', name), ...
            logger, true, 'log_scriptinfo');
                                
        % do scripting
        funcname = prepare_runfunc(curscript.func, rootdir);
        
        scriptpath = sladdpath(curscript.path, rootdir);
        ctrlpath = curscript.ctrlpath;
        refs = load_references(name, curscript, rootdir);
        workdir = curscript.params.workdir;
        
        props = feval(funcname, refs, curscript.params);
        edl_writescript(scriptpath, gidstr, workdir, ctrlpath, props);
                                
        write(logger, 'Finishing scripting');
        writeblank(logger);
            
        
    case 'experiment'
                
        scope = 'experiments';
        curexp = take_element(expdef, scope, name);
        
        % output experiment
                
        init_instance(curexp, ...
            sprintf('Starting experiment %s', name), ...
            logger, false, 'log_expinfo');
        
        % do experiment
        
        funcname = prepare_runfunc(curexp.func, rootdir);    
        scriptpath = sladdpath(curexp.scriptpath, rootdir);
        
        edl_batchexp(funcname, scriptpath, expdef.env, logger, filter, runopt);
        
        write(logger, 'Finishing experiment');
        writeblank(logger);
        
    case 'report'
        
        scope = 'reports';
        curreport = take_element(expdef, scope, name);
        
        % output script info
        
        gidstr = init_instance(curreport, ...
            sprintf('Starting reporting %s', name), ...
            logger, true, 'log_reportinfo');
                                
        % do reporting
        funcname = prepare_runfunc(curreport.func, rootdir); 
        refs = load_references(name, curreport, rootdir);
        reportpath = sladdpath(curreport.path, rootdir);
        
        props = feval(funcname, refs, curreport.params, expdef.env);
        edl_writereport(reportpath, gidstr, props);
                                
        write(logger, 'Finishing reporting');
        writeblank(logger);
                        
    otherwise
        error('Invalid action type for EDL: %s', type);
end



%% Information Loggings

function log_scriptinfo(s, logger)

writeinfo(logger, 'function = %s', s.func);
writeinfo(logger, 'script path = %s', s.path);
writeinfo(logger, 'control path = %s', s.ctrlpath);
log_reftable(s, logger);
log_params(s.params, logger);

function log_reportinfo(s, logger)

writeinfo(logger, 'function = %s', s.func);
writeinfo(logger, 'script path = %s', s.path);
log_reftable(s, logger);
log_params(s.params, logger);

function log_expinfo(s, logger)

writeinfo(logger, 'function = %s', s.func);
writeinfo(logger, 'script path = %s', s.scriptpath);


function log_reftable(s, logger)

rps = s.refpaths;

if ~isempty(rps)
    writeinfo(logger, 'references:');
    logger = incindent(logger, 1);
    
    roles = fieldnames(rps);
    n = length(roles);    
    for i = 1 : n
        role = roles{i};
        writeinfo(logger, '%s = %s', role, rps.(role));
    end
end


%% Auxiliary functions

function funcname = prepare_runfunc(funcpath, rootdir)

funcdir = sladdpath(slfilepart(funcpath, 'parent'), rootdir);
funcname = slfilepart(funcpath, 'title');
cd(funcdir);

function R = take_element(expdef, scope, name)

sset = expdef.(scope);
if ~isfield(sset, name)
    error('edl:interperror', ...
        'The specified name %s is not found in %s of the experiment definition', ...
        name, scope);
end
R = sset.(name);

function gidstr = init_instance(S, startmsg, logger, genguid, funcloginfo)

if genguid
    gidstr = slguidstr(slguid);
    writeinfo(logger, '[%s]', gidstr);
else
    gidstr = '';
end

if ~isempty(startmsg)
    write(logger, startmsg);
else
    write(logger, 'Start Instance');
end

logger = incindent(logger, 1);

if ~isempty(funcloginfo)
    feval(funcloginfo, S, logger);
end

writeblank(logger);


function refs = load_references(name, elem, rootdir)

if isempty(elem.refpaths)
    refs = [];
else
    fns = fieldnames(elem.refpaths);
    n = length(fns);
    for i = 1 : n
        fn = fns{i};
        
        curscope = elem.refscopes.(fn);
        curpath = sladdpath(elem.refpaths.(fn), rootdir);
        
        switch curscope
            case 'scripts'
                curref = edl_readscript(curpath);
            case 'reports'
                curref = edl_readreport(curpath);
            otherwise
                error('edl:interperror', ...
                    'Invalid scope name %s for item %s in %s', ...
                    curscope, fn, name);                
        end
        
        refs.(fn) = curref;        
    end
    
end

function log_params(params, logger)

if ~isempty(params)
    fns = fieldnames(params);
    n = length(fns);
    writeinfo(logger, 'parameters: ');
    logger = incindent(logger, 1);

    for i = 1 : n
        fn = fns{i};
        writeinfo(logger, '%s = %s', fn, params.(fn));
    end

end





