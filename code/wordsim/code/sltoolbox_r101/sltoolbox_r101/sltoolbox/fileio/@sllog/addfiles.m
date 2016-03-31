function logger = addfiles(logger, files, initstatus)
%ADDFILES add a set of log files to the logger
%
% $ Syntax $
%   - logger = addfiles(logger, file)
%   - logger = addfiles(logger, files)
%   - logger = addfiles(logger, files, initstatus)
%
% $ Arguments $
%   - logger:       the logger you want to add files to
%   - file:         a string of a filename
%   - files:        a cell array of filenames
%   - initstatus:   whether is activated initially (default = true)
%
% $ Description $
%   - logger = addfiles(logger, file) adds one file to logger
%   
%   - logger = addfiles(logger, files) adds a set of files to logger
%
%   - logger = addfiles(logger, files, initstatus) additionally sets
%     the initial states of the added files.
%
% $ Remarks $
%   - Repeated filename cannot be added to the logger list.
%
%   - The files will be open before adding, and the status will be
%     initially set to true by default.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12th, 2006
%

if nargin < 3
    initstatus = true;
else
    if ~islogical(initstatus) || numel(initstatus) ~= 1
        error('sltoolbox:invalidarg', ...
            'The initstatus should be a logical variable');
    end
end


if ischar(files)
        
    if isattached(logger, files)
        error('sltoolbox:invalidarg', ...
            'The file %s has been added to the file list', fp);
    end
                      
    fp = sladdpath(files, logger.rootpath);
    fid = fopen(fp, 'at');
    if fid <= 0
        error('sltoolbox:fileioerror', ...
            'Fail to open log file %s', fp);
    end
    
    curn = length(logger.files);
    idx = curn + 1;
    logger.files(idx, 1).filepath = fp;
    logger.files(idx, 1).fid = fid;
    logger.files(idx, 1).isactive = initstatus;
    
elseif iscell(files)
    
    n = numel(files);
    for i = 1 : n
        logger = addfiles(logger, files{i}, initstatus);
    end
    
else
    error('sltoolbox:invalidarg', ...
        'The files should be either a filename or a cell array of filenames');
end

    
    
    
    