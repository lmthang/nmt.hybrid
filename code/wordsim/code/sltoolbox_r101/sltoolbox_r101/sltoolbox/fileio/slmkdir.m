function dirpath = slmkdir(dirname, parentpath)
%SLMKDIR Makes a directory if it does not exist
%
% $ Syntax $
%   - dirpath = slmkdir(dirname)
%   - dirpath = slmkdir(dirname, parentpath)
%
% $ Arguments $
%   - dirname:      the name (relative path) of the diretory (r.t parent)
%   - parentpath:   the path of the parent directory
%   - dirpath:      the path of the created directory (r.t. '')
%
% $ Description $
%   - dirpath = slmkdir(dirname) makes a diretory of given name if it
%     does not exist.
%
%   - dirpath = slmkdir(dirname, parentpath) makes a diectory of given 
%     name if it does not exist, relative to parent path.
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
if ~exist(dirpath, 'dir')
    mkdir(dirpath);
end

