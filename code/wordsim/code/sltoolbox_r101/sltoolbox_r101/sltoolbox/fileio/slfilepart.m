function s = slfilepart(fp, partname)
%SLFILEPARTS Extracts a specified part of a file path string
%
% $ Syntax $
%   - s = slfilepart(fp, partname)
%
% $ Arguments $
%   - fp:           the path string
%   - partname:     the name of the querying part
%   - s:            the string of that part
%
% $ Description $
%   - s = slfilepart(fp, partname) extracts the corresponding part.
%     The partname can be either of the following:
%       - 'name':   title.ext
%       - 'ext':    .ext
%       - 'title'   title
%       - 'parent'  parent path string
%     
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

if nargin < 2
    raise_lackinput('slfilepart', 2);
end

[p.parent, p.title, p.ext] = fileparts(fp);

switch partname
    case 'name'
        s = [p.title, p.ext];
    case 'ext'
        s = p.ext;
    case 'title'
        s = p.title;
    case 'parent'
        s = p.parent;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid part name for a path: %s', partname);
end
