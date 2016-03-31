function slrmdir(dirname, parentpath)
%SLMKDIR Makes a directory if it does not exist
%
% $ Syntax $
%   - slrmdir(dirname)
%   - slrmdir(dirname, parentpath)
%
% $ Arguments $
%   - dirname:      the name (relative path) of the diretory (r.t parent)
%   - parentpath:   the path of the parent directory
%   - dirpath:      the path of the created directory (r.t. '')
%
% $ Description $
%   - dirpath = slmkdir(dirname) deletes a diretory of given name if it
%     exists.
%
%   - dirpath = slmkdir(dirname, parentpath) deletes a diectory of given 
%     name if it exists, relative to parent path.
%
% $ Remarks $
%   - This function recursively deletes all sub-folders.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%  

if isempty(dirname)
    return;
end

if nargin < 2 
    parentpath = '';
end

dirpath = sladdpath(dirname, parentpath);

if exist(dirpath, 'dir')
    rmdir(dirpath, 's');
end
    



