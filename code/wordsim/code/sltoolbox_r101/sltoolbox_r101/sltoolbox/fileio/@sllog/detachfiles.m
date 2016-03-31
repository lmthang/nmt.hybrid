function logger = detachfiles(logger, files)
%DETACHFILES Detachs the logger from some added files
%
% $ Syntax $
%   - logger = detachfiles(logger, file)
%   - logger = detachfiles(logger, files)
%
% $ Argument $
%   - logger:       the logger to be processed
%   - file:         the filename of the file to be detached
%   - files:        the cell array of filenames to be detached
%
% $ Description $
%   - logger = detachfiles(logger, file) detach a file from the logger
%     list.
%
%   - logger = detachfiles(logger, files) detach a set of filenames from
%     the logger list.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

if ischar(files)
        
    [tf, idx] = isattached(logger, files);
    if ~tf
        error('sltoolbox:invalidarg', ...
            'The file %s has not been added to the file list', fp);
    end
    
    fclose(files(idx).fid);
    
elseif iscell(files)
    
    n = numel(files);
    for i = 1 : n
        logger = detachfiles(logger, files{i});
    end
    
else
    error('sltoolbox:invalidarg', ...
        'The files should be either a filename or a cell array of filenames');
end
    
    
    