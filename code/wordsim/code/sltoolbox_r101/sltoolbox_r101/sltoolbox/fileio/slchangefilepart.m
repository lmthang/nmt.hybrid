function newfp = slchangefilepart(fp, varargin)
%SLCHANGEFILEPART Changes some parts of the file path
%
% $ Syntax $
%   - newfp = slchangefilepart(fp, partname1, part1, ...)
%
% $ Description $
%   - newfp = slchangefilepart(fp, partname1, part1, ...) changes the 
%     specified part of a path to a new value to form a new path.
%     Please refer to slfilepart for part names
%
% $ Remarks $
%   - If you specify the name, then you should be specify title and ext.
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

%% Main body

if isempty(varargin)
    newfp = fp;
else
    
    opts.parent = '';
    opts.name = '';
    opts.title = '';
    opts.ext = '';
    opts = slparseprops(opts, varargin{:});
    
    if ~isempty(opts.ext)
        if opts.ext(1) ~= '.'
            error('sltoolbox:invalidarg', ...
                'The extension string should start with a dot . ');
        end
    end
    
    [p.parent, p.title, p.ext] = fileparts(fp);
    
    if isempty(opts.name)
        p = updatefields(p, opts, {'parent', 'title', 'ext'});
        newfp = fullfile(p.parent, [p.title, p.ext]);
    else        
        if ~isempty(opts.title) || ~isempty(opts.ext)
            error('sltoolbox:invalidarg', ...
                'When name is specified, title and ext should not be');
        end
        p = updatefields(p, opts, {'parent', 'name'});
        newfp = fullfile(p.parent, p.name);
    end
        
end


%% Auxiliary functions

function S = updatefields(S, newS, fns)

nf = length(fns);
for i = 1 : nf
    f = fns{i};
    if ~isempty(newS.(f))
        S.(f) = newS.(f);
    end
end

    





    
    