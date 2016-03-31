function logger = sllog(varargin)
%SLLOG Constructs a logger
%
% $ Syntax $
%   - logger = sllog(...)
%
% $ Description $
%   - logger = sllog(...) constructs a logger using the specified 
%     properties.
%     \*
%     \t    The Logger Properties
%     \h    name       &    description           \\
%          'rootpath'   & The root path of the log filename, default = ''
%          'files'      & the cell array of log filenames, default = {}
%          'winshow'    & whether to show to matlab window, default = true
%          'timestamp'  & whether to add time-stamp to records, 
%                         default = true
%          'timeformat' & the format string for time-stamp
%                         can be either a number or a string
%                         refer to datastr for detail.
%                         default = '[yyyy-mm-dd HH:MM:SS]'
%     \*
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

%% parse and verify input arguments

opts.rootpath = '';
opts.files = {};
opts.winshow = true;
opts.timestamp = true;
opts.timeformat = '[yyyy-mm-dd HH:MM:SS]';

opts = slparseprops(opts, varargin{:});

%% construction

logger = struct(...
    'rootpath', opts.rootpath, ...
    'files', [], ...
    'winshow', opts.winshow, ...
    'indent', 0, ...
    'timestamp', true, ...
    'timeformat', opts.timeformat ...
    );
logger = class(logger, 'sllog');

%% open file handles

logger = addfiles(logger, opts.files);


