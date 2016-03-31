function logger = setactive(logger, files, s)
%SETACTIVE Sets a set of files active/inactive
%
% $ Syntax $
%   - logger = setactive(logger, file, s)
%   - logger = setactive(logger, files, s)
%
% $ Description $
%   - logger = setactive(logger, file, s) sets the state of the specified
%     file in the logger to be s (true/false).
%
%   - logger = setactive(logger, file, s) sets the state of the specified
%     cell array of files in the logger to be s (true/false).
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

if ischar(files)
        
    [tf, idx] = isattached(logger, files);
    
    if tf
        logger.files(idx).isactive = s;
    else
        error('sltoolbox:invalidarg', ...
            'The file %s has not been added to the file list', files);
    end
    
elseif iscell(files)
    
    n = length(files);
    for i = 1 : n
        logger = setactive(logger, files{i}, s);
    end
    
else
    
    error('sltoolbox:invalidarg', ...
        'The files should be a string or a cell array');
end

    
