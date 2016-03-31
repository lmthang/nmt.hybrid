function h = sldrawpts(X, varargin)
% SLDRAWPTS Draws a set of sample points on axes
%
% $ Syntax $
%   - h = sldrawpts(X, ...)
%   - h = sldrawpts(X, plotsyms, ...)
%   - h = sldrawpts(X, nums, ...)
%   - h = sldrawpts(X, nums, plotsyms, ...)
%
% $ Arguments $
%   - X:        the sample matrix with each column as a sample
%   - nums:     the number of samples in classes
%   - plotsyms: the cell array of plot symbols for classes
%               (must be encompassed in a cell array)
%   - h:        the a column vector of handles to lineseries objects, 
%               one handle per plotted line. 
%     
% $ History $
%   - Created by Dahua Lin, on Aug 25, 2006
%   - Modified by Dahua Lin, on Aug 28, 2006
%       - fix the error when nums have zeros
%

%% parse and verify input arguments

[d, n] = size(X);
if d ~= 2 && d ~= 3
    error('sltoolbox:invalidarg', ...
        'X should contain 2D or 3D samples');
end

plotprops = {};

if nargin == 1
    multiclass = false;
    plotsyms = default_plotsyms(1);
    plotprops = {};
    
else
    if ~isnumeric(varargin{1})  % single class
        multiclass = false;
        if ~iscell(varargin{1})
            plotsyms = default_plotsyms(1);
            pidx = 1;
        else
            plotsyms = varargin{1};
            pidx = 2;
        end
        
    else                        % multi class                     
        multiclass = true;
        nums = varargin{1};
        nums = nums(:)';
        if sum(nums) ~= n
            error('sltoolbox:sizmismatch', ...
                'The nums are inconsistent with the number of samples');
        end
        
        c = length(nums);
        if nargin == 2 || ~iscell(varargin{2})
            plotsyms = default_plotsyms(c);
            pidx = 2;
        else
            plotsyms = varargin{2};
            pidx = 3;
        end
        
    end
        
    if pidx <= length(varargin)
        plotprops = varargin(pidx:end);
    end
        
end


%% Argument Preparation

if ~multiclass
    if d == 2        
        args = {X(1,:), X(2,:), plotsyms{1}};
    elseif d == 3
        args = {X(1,:), X(2,:), X(3,:), plotsyms{1}};
    end            
else
    [sinds, einds] = slnums2bounds(nums);
    ck = 0;
    
    if d == 2
        args = cell(3, sum(nums > 0));
        for k = 1 : c
            if nums(k) > 0
                ck = ck + 1;
                si = sinds(k); ei = einds(k);
                args{1, ck} = X(1, si:ei);
                args{2, ck} = X(2, si:ei);
                args{3, ck} = plotsyms{k};
            end                        
        end
        
    elseif d == 3
        args = cell(4, sum(nums > 0));
        for k = 1 : c
            if nums(k) > 0
                ck = ck + 1;
                si = sinds(k); ei = einds(k);
                args{1, ck} = X(1, si:ei);
                args{2, ck} = X(2, si:ei);
                args{3, ck} = X(3, si:ei);
                args{4, ck} = plotsyms{k};
            end
        end
        
    end
    args = reshape(args, [1, numel(args)]);
    
end

%% Plot

if nargout == 0
    if d == 2
        plot(args{:}, plotprops{:});
    elseif d == 3
        plot3(args{:}, plotprops{:});
    end
else
    if d == 2
        h = plot(args{:}, plotprops{:});
    elseif d == 3
        h = plot3(args{:}, plotprops{:});
    end
end


%% Auxiliary functions

function plotsyms = default_plotsyms(n)

ps0 = {'b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.'};
n0 = length(ps0);

inds = mod(0:n-1, n0) + 1;
plotsyms = ps0(inds);


