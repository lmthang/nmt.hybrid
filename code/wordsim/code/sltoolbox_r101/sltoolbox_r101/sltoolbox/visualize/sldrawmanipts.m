function [h, cm] = sldrawmanipts(X, s, crspec)
%SLDRAWMANIPTS Draws the sample points on a manifold
%
% $ Syntax $
%   - sldrawmanipts(X)
%   - sldrawmanipts(X, s, crspec)
%   - h = sldrawmanipts(X, s, crspec)
%   - [h, cm] = sldrawmanipts(X, s, crspec)
%
% $ Arguments $
%   - X:        The sample matrix (2 x n or 3 x n)
%   - s:        The scalar indicating the marker size (default = 5)
%   - crspec:   The specification of the colors, can be in either of the
%               following forms:
%               - a string (plot symbol) indicating a basic color
%               - a 1 x n row vector as a color map
%               - a function handle to calculate the color map
%                   c = f(x, y) for 2D samples
%                   c = f(x, y, z) for 3D samples
%               default = 'b';
%   - h:        The handles to the drawn objects
%   - cm:       The color map used
% 
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006   
%               

%% parse and verify arguments

if ndims(X) ~= 2 
    error('sltoolbox:invalidarg', ...
        'The sample matrix X should be a 2D matrix');
end

[d, n] = size(X);
if d ~= 2 && d ~= 3
    error('sltoolbox:invalidarg', ...
        'The samples should be 2D or 3D');
end
if d == 2
    x = X(1, :);
    y = X(2, :);
else
    x = X(1, :);
    y = X(2, :);
    z = X(3, :);
end

if nargin < 2 || isempty(s)
    s = 5;
else    
    if ~isscalar(s)
        error('sltoolbox:invalidarg', ...
            's should be a scalar');
    end
end

if nargin < 3 || isempty(crspec)
    cm = 'b';
else
    if ischar(crspec)
        cm = crspec;
    elseif isnumeric(crspec)
        if ~isequal(size(crspec), [1,n])
            error('sltoolbox:sizmismatch', ...
                'The size of the color map is not consistent with the sample number');
        end
        cm = crspec;
    elseif isa(crspec, 'function_handle')
        if d == 2
            cm = crspec(x, y);
        else
            cm = crspec(x, y, z);
        end
    else
        error('sltoolbox:invalidarg', ...
            'crspec should be a string, a row vector or a function handle');
    end
end

%% draw

if d == 2
    if nargout == 0
        scatter(x, y, s, cm);
    else
        h = scatter(x, y, s, cm);
    end
else
    if nargout == 0
        scatter3(x, y, z, s, cm);
    else
        h = scatter3(x, y, z, s, cm);
    end
end
        
    






