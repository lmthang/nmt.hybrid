function h = sldrawmultiellipse(centers, vars, npts, plotsyms, varargin)
%SLDRAWMULTIELLIPSE Draws multiple ellipses on axies
%
% $ Syntax $
%   - sldrawmultiellipse(centers, vars, npts)
%   - sldrawmultiellipse(centers, vars, npts, plotsyms, ...)
%   - h = sldrawmultiellipse(...)
%
% $ Arguments $
%   - centers:      the centers of the ellipse 
%   - vars:         the variances/covariances of the ellipses
%   - npts:         the number of sample points on each ellipses
%                   (default = 300)
%   - plotsyms:     the cell array of plot symbols charcterizing the
%                   ellipses. It can be a char string, representing
%                   the symbol shared by all ellipses, or a cell array
%                   of symbols for different ellipses.
%   - h:            the column array of numbers identifying the plots
%
% $ Description $
%    - sldrawmultiellipse(centers, vars, npts) plots one or multiple
%      ellipses on the same axis. The ellipses are charcterized by 
%      gaussian models parameterized by their centers and variances/
%      covariances. The plots are based on sampling points on the 
%      ellipse with Mahalanobis distance being 1. The number of samples
%      are specified through npts. 
%      If there are k ellipses, centers should be a 2 x k matrix, while
%      the vars can be either of the following forms:
%           - 1 x k matrix:     isotropic variance 
%           - 2 x k matrix:     diagonal variance
%           - 2 x 2 x k array:  normal covariance
%      If the variances/covariances are shared by all ellipses, they
%      can be encapsulated in a cell.
%           - {1 x 1 scalar}:   shared isotropic variance
%           - {2 x 1 vector}:   shared diagonal variance
%           - {2 x 2 matrix}:   shared covariance matrix
%           
% $ Remarks $
%   - It is based on sldrawellipse for plotting.
%
% $ History $
%   - Created by Dahua Lin, on Aug 26, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sldrawmultiellipse', 2);
end

if ndims(centers) ~= 2 || size(centers, 1) ~= 2
    error('sltoolbox:invalidarg', ...
        'centers should be a 2 x k matrix');
end

k = size(centers, 2);

% parse variance/covariance

if isnumeric(vars)
    if isequal(size(vars), [1 k])
        varform = 1;
    elseif isequal(size(vars), [2 k])
        varform = 2;
    elseif isequal(size(vars), [2 2 k])
        varform = 3;
    else
        error('sltoolbox:sizmismatch', ...
            'The size of vars is illegal');
    end
    sharevar = false;
elseif iscell(vars) && numel(vars) == 1
    vars = vars{1};
    if isequal(size(vars), [1 1])
        varform = 1;
    elseif isequal(size(vars), [2 1])
        varform = 2;
    elseif isequal(size(vars), [2 2])
        varform = 3;
    else
        error('sltoolbox:sizmismatch', ...
            'The size of vars is illegal');
    end
    sharevar = true;
end

if nargin < 3 || isempty(npts)
    npts = 300;
end

if nargin < 4 || isempty(plotsyms)
    plotsyms = dme_default_plotsyms(k);
else
    if ischar(plotsyms)
        ch = plotsyms;
        plotsyms = cell(1, k);
        [plotsyms{:}] = deal(ch);
    elseif iscell(plotsyms)
        if length(plotsyms) ~= k
            error('sltoolbox:sizmismatch', ...
                'The length of plotsyms is inconsistent with the number of ellipses');
        end
    else
        error('sltoolbox:sizmismatch', ...
            'The plotsyms should be either a string or a cell array');
    end
end

if nargin < 5
    plotprops = {};
else
    plotprops = varargin;
end

if nargout >= 1
    outh = true;
    h = zeros(k, 1);
else
    outh = false;
end

%% Main body

for i = 1 : k
    vcenter = centers(:, i);
    mcov = dme_get_cov(vars, i, varform, sharevar);
    curh = sldrawellipse(vcenter, mcov, npts, plotsyms{i}, plotprops{:});
    if outh 
        h(i) = curh;
    end
end


%% Auxiliary functions

function r = dme_default_plotsyms(n)

pss = {'b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-'};
inds = mod(0:n-1, 7) + 1;
r = pss(inds);

function C = dme_get_cov(vars, i, varform, sharevar)

switch varform 
    case 1
        if sharevar
            C = [vars 0; 0 vars];
        else
            C = [vars(i) 0; 0 vars(i)];
        end
    case 2
        if sharevar
            C = diag(vars);
        else
            C = diag(vars(:, i));
        end
    case 3
        if sharevar
            C = vars;
        else
            C = vars(:,:,i);
        end            
end
    
    








    
    
    