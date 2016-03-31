function R = get(logger, propname)
%GET Gets properties of a logger
%
% $ Syntax $
%   - R = get(logger, propname)
%
% $ Description $
%   - R = get(logger, propname) gets the property of specified name for
%     the logger.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

switch propname
    case 'rootpath'
        R = logger.rootpath;
    case 'indent'
        R = logger.indent;
    case 'numfiles'
        R = length(logger.files);
    case 'filepaths'
        R = {logger.files.filepath}';
    case 'winshow'
        R = logger.winshow;
    case 'files'
        R = logger.files;
    case 'timestamp'
        R = logger.timestamp;
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid property name %s for sllog', propname);
end
