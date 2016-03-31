function b = isactive(logger, filename)
%ISACTIVE Queries whether a log file is active
%
% $ Syntax $
%   - b = isactive(logger, filename)
%
% $ Description $
%   - b = isactive(logger, filename) queries whether a log file is active.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12, 2006
%

[tf, idx] = isattached(logger, filename);

if tf
    b = logger.files(idx).isactive;
else
    error('sltoolbox:invalidarg', ...
            'The file %s has not been added to the file list', fp);
end

