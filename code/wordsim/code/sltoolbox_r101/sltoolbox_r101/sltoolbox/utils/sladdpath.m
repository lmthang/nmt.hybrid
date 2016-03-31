function paths = sladdpath(filenames, dirpath)
%SLADDPATH Adds dirpath to precede the filenames
%
% $ Syntax $
%   - paths = sladdpath(filenames, dirpath)
%
% $ Arguments $
%   - filenames:        the filenames without root path
%   - dirpath:          the preceding dirpath to be added
%   - paths:            the full names after the dirpath is added
%
% $ Description $
%   - paths = sladdpath(filenames, dirpath) adds directory path in front
%     of the filenames to form full paths.
%
% $ History $
%   - Created by Dahua Lin, on Jul 27th, 2006
%

%% Parse and verify input arguments

if nargin < 2
    raise_lackinput('sladdpath', 2);
end

if iscell(filenames)
    ismulti = true;
elseif ischar(filenames)
    ismulti = false;
else
    error('sltoolbox:invalidarg', ...
        'The filenames should be either a single char string of the filename or a cell array of strings');
end


%% Main skeleton

if ~ismulti
    if ~isempty(dirpath)
        paths = internal_addpath(filenames, dirpath);
    else
        paths = filenames;
    end
else
    if ~isempty(dirpath)
        n = numel(filenames);
        paths = cell(size(filenames));
        for i = 1 : n
            paths{i} = internal_addpath(filenames{i}, dirpath);
        end
    else
        paths = filenames;
    end
end


%% Sub-functions

function pathstr = internal_addpath(filename, dirpath)
% precondition: dirpath is not empty

if dirpath(end) ~= '\'
    pathstr = [dirpath, '\', filename];
else
    pathstr = [dirpath, filename];
end


    








 