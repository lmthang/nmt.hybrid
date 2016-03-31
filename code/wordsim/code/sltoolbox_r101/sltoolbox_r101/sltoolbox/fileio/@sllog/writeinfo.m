function writeinfo(logger, varargin)
%WRITEINFO Writes information to logger without time-stamp
%
% $ Syntax $
%   - writeinfo(logger, ...)
%
% $ Description $
%   - writeinfo(logger, ...). The usage is the same as write, except that
%     the time stamp becoming blank during writing.
%
% $ History $
%   - Created by Dahua Lin, on Aug 13, 2006
%

if logger.timestamp
    tstr = datestr(now(), logger.timeformat);
    logger.timeformat = blanks(length(tstr));
    write(logger, varargin{:});
else
    write(logger, varargin{:});
end

