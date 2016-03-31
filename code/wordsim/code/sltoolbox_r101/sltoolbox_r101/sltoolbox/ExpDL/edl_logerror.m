function edl_logerror(caller, err, logger, varargin)
%EDL_LOGERROR Logs an error into logger
%
% $ Syntax $
%   - edl_logerror(caller, err, logger, ...)
%
% $ Arguments $
%   - caller:       the name of caller (the agent that catches the error)
%   - err:          the error struct
%   - logger:       the logger
% 
% $ Description $
%   - edl_logerror(caller, err, logger, ...) logs the error caught into
%     a logger. The logged information may include error header, 
%     message, and runtime stack, depending on following properties
%       'header':   whether to log header info (default = true)
%       'message':  whether to log error message (default = true)
%       'stack':    whether to log run-time stack (default = true)
%
% $ History $
%   - Created by Dahua Lin, on Aug 13, 2006
%

%% Parse and verify input arguments

if nargin < 3
    raise_lackinput('edl_logerror', 3);
end

opts.header = true;
opts.message = true;
opts.stack = true;
opts = slparseprops(opts, varargin{:});



%% Logging

% log header

if opts.header
    if isempty(err.identifier)
        write(logger, '%s catch error:', caller);
    else
        write(logger, '%s catch error: %s:', caller, err.identifier);
    end
end

% log message

if opts.message    
    msglines = slstrsplit(err.message, sprintf('\r\n'));
    nmlines = length(msglines);
    for i = 1 : nmlines
        writeinfo(logger, '%s', msglines{i});
    end
    writeblank(logger);
end

% log stack

if opts.stack
    if ~isempty(err.stack)
        es = err.stack;
        writeinfo(logger, 'Runtime Stack:');
        ns = length(es);
        logger = incindent(logger, 1);
        for k = 1 : ns
            if ~isempty(es(k).file)
                writeinfo(logger, '%s ==> (%d at %s)', es(k).name, es(k).line, es(k).file);
            else
                writeinfo(logger, '%s ==> (%d)', es(k).name, es(k).line);
            end
        end
    end
    writeblank(logger);
end

