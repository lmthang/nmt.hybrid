function write(logger, varargin)
%WRITE Writes message to a logger
%
% $ Syntax $
%   - write(logger, ...)
%
% $ Description $
%   - write(logger, ...) writes message to a logger. The contents given
%     should be as sprintf.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%


if ~logger.winshow && isempty(logger.files)
    return;
end

% generate string

indent_step = 4;    
ns = max(logger.indent, 0);

if logger.timestamp
    tstr = datestr(now(), logger.timeformat);
    nblanks = ns * indent_step + 2;
    str = [tstr, blanks(nblanks), sprintf(varargin{:})];
else
    if ns == 0
        str = sprintf(varargin{:});
    else
        nblanks = ns * indent_step;
        str = [blanks(nblanks), sprintf(varargin{:})];
    end
end

% write it

if logger.winshow
    disp(str);
end

if ~isempty(logger.files)
    
    F = logger.files;
    nf = length(F);
    for i = 1 : nf        
        if F(i).isactive
            fprintf(F(i).fid, '%s\n', str);
        end        
    end
        
end
    

    
    
    
    

