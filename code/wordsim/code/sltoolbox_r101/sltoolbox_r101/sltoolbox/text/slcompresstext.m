function Tc = slcompresstext(T, varargin)
%SLCOMPRESSTEXT Compresses a cell array of text
%
% $ Syntax $
%   - Tc = slcompresstext(T, ...)
%
% $ Arguments $
%   - T:        the source text to be compressed
%   - Tc:       the compressed text
%
% $ Description $
%   - Tc = slcompresstext(T, ...) compresses the text T, represented in
%     cell array of lines. You can specify the properties to control the
%     process of compression.
%     \*
%     \t    Table 1. Properties of Text Compression
%     \h    name        &    description
%          'rmempty'    &  whether to remove empty line (default = true)
%          'proc'       &  the method of processing each line
%                          (default='trim')
%                          - 'off': do not process
%                          - 'trim': trim the leading and trailing spaces
%                          - 'deblank': trim only the trailing spaces
%
% $ History $
%   - Created by Dahua Lin, on Aug 9th, 2006
%

%% parse and verify input arguments

if ~iscell(T)
    error('sltoolbox:invalidarg', ...
        'T should be a cell array of strings');
end

opts.rmempty = true;
opts.proc = 'trim';
opts = slparseprops(opts, varargin{:});

switch opts.proc
    case 'off'
        procfunc = [];
    case 'trim'
        procfunc = 'strtrim';
    case 'deblank'
        procfunc = 'deblank';
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid string processing method %s', opts.proc);
end

%% Process

if ~isempty(procfunc)
    Tc = slfiltertext(T, procfunc);
else
    Tc = T;
end

%% Select

if opts.rmempty
    nlines = length(Tc);
    is_effline = true(nlines, 1);
    has_deleted = false;
    for i = 1 : nlines
        if isempty(Tc{i})
            is_effline(i) = false;
            has_deleted = true;
        end        
    end
    
    if has_deleted
        Tc = Tc(is_effline);
    end
end





