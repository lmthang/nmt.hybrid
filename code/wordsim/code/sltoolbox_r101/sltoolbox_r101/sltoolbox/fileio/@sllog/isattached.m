function [tf, idx] = isattached(logger, filename)
%ISATTACHED Judges whether the file is attached to the logger
%
% $ Syntax $
%   - [tf, idx] = isattached(logger, filename)
%
% $ Arguments $
%   - logger:       the logger to be queried
%   - filename:     the filename to be queried
%   - tf:           the boolean variable indicating whether attach
%   - idx:          the index of the found file
%
% $ Description $
%   - [tf, idx] = isattached(logger, filename) judges whether a file
%     is currently attached to the logger. The filename should be 
%     given as relative path to the logger's root path.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

tf = false;
idx = 0;

if ~isempty(logger.files)
    
    fp = sladdpath(filename, logger.rootpath);
    
    n = length(logger.files);
    for i = 1 : n        
        if strcmpi(fp, logger.files(i).filepath)
            tf = true;
            idx = i;
            break;
        end        
    end
    
end
