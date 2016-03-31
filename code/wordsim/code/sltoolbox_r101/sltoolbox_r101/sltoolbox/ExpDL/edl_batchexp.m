function edl_batchexp(expfun, scrpath, env, logger, filter, runopt)
%EDL_BATCHEXP Performs Batch experiments according to scheme
%
% $ Syntax $
%   - edl_batchexp(expfun, scrpath, env, logger)
%   - edl_batchexp(expfun, scrpath, env, logger, filter, runopt)
%
% $ Arguments $
%   - expfun:   the experiment function
%   - scrpath:  the absolute script path
%   - env:      the environment configuration
%   - logger:   the logger to log experiment information
%               (if not specified, it will create a default one)
%   - filter:   the filtering function (default = [])
%               it can also be a integer array specifying which 
%               experiments to run.
%   - runopt:   the running option
%               'restart':  run all specified experiments 
%                           (default follows edl_batchexp)
%               'resume':   run the experiments that not succeeded
%
% $ Description $
%   - edl_batchexp(expfun, scrpath, env) performs a batch of experiments 
%     according to the properties list in sch and the environment variables. 
%
%   - edl_batchexp(expfun, scrpath, env, filter) additionaly uses a filter 
%     function to select a subset of experiments from sch to run. The filter
%     function should receive one argument as the property struct, and
%     outputs true or false.
%
% $ History $
%   - Created by Dahua Lin, on Aug 10th, 2006
%   - Modified by Dahua Lin, on Aug 13rd, 2006
%       - Based on new EDL specification
%

%% Parse and verify input arguments

if nargin < 3
    raise_lackinput('edl_batchexp', 3);
end

if nargin < 4 || isempty(logger)
    logger = sllog();
end

if nargin < 5
    filter = [];
end

if nargin < 6 || isempty(runopt)
    runopt = 'restart';
end

     

%% Main Skeleton

% prepare configuration
[copts, sch] = prepare_configuration(scrpath, logger, filter, runopt);

% do experiments
do_experiments(expfun, copts, sch, env, logger);

% finalize
write(logger, 'Experiments completed.');


%% Core routines

function [copts, sch] = prepare_configuration(scrpath, logger, filter, runopt)

switch runopt
    case 'restart'
        is_resume = false;
    case 'resume'
        is_resume = true;
    otherwise
        error('edl:interperror', ...
            'Invalid running option %s', runopt);
end


write(logger, 'Prepare experiment configurations');

script = edl_readscript(scrpath);
scrparent = slfilepart(scrpath, 'parent');

copts.guid = script.attribs.guid;
copts.workdir = script.attribs.workdir;
copts.ctrlpath = sladdpath(script.attribs.ctrlpath, scrparent);

logger = incindent(logger, 1);
writeinfo(logger, 'GUID = %s', copts.guid);
writeinfo(logger, 'workdir root = %s', copts.workdir);
writeinfo(logger, 'control path = %s', copts.ctrlpath);

sch = script.entries;

if ~isempty(filter)
    writeinfo(logger, 'filtered scheme = true');
    sch = filter_sch(sch, filter);
end

writeinfo(logger, 'running mode = %s', runopt);
if is_resume
    sch = filter_sch_runopt(sch, copts.ctrlpath);
end

writeinfo(logger, 'number of experiments: %d', length(sch));
writeblank(logger);


function sch = filter_sch(sch, filter)

n = length(sch);

is_selected = false(n, 1);

if isnumeric(filter)
    is_selected(filter) = true;
else
    for i = 1 : n
        is_selected(i) = feval(filter, sch(i));
    end
end

if ~all(is_selected)
    sch = sch(is_selected);
end

function sch = filter_sch_runopt(sch, ctrlpath)

n = length(sch);

if n > 0
    C = edl_readctrlfile(ctrlpath);
    is_disabled = false(n, 1);
    
    for i = 1 : n
        idx = sch(i).internal_index;
        if strcmpi(C.status{idx}, 'succeed')
            is_disabled(i) = true;
        end        
    end
end


if any(is_disabled)
    sch = sch(~is_disabled);
end



function do_experiments(expfun, copts, sch, env, logger)

n = length(sch);

for i = 1 : n
    
    cursch = sch(i);
    internal_index = cursch.internal_index;
    
    write(logger, 'Enter experiment %d / %d (internal index = %d)', i, n, internal_index);
    
    try
        feval(expfun, ...
            cursch, ...
            env, ...
            incindent(logger, 1));
        
        edl_updatectrlfile(copts.guid, copts.ctrlpath, internal_index, 'succeed');
        write(logger, 'experiment succeeded');
        writeblank(logger);
        
    catch        
        write(logger, 'experiment failed');                
        writeblank(logger);
        edl_updatectrlfile(copts.guid, copts.ctrlpath, internal_index, 'failed');
        
        rethrow(lasterror);
    end
            
end


