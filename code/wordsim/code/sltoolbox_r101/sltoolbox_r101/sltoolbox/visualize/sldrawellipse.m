function h = sldrawellipse(center, shape, n, varargin)
%SLDRAWELLIPSE Draws an ellipse on current axis
%
% $ Syntax $
%   - sldrawellipse(center, shape)
%   - sldrawellipse(center, shape, n)
%   - sldrawellipse(center, shape, n, ...)
%   - h = sldrawellipse(...)
%
% $ Description $
%   - sldrawellipse(center, shape) draws an ellipse specified by
%     its center and shape on current axis. center is a vector of
%     length 2, storing the x and y coordinate of the center. 
%     shape have following two forms:
%     1. [a, b, theta], here a and b are the half-lengths of two axis,
%        theta is the radians of first axis relative to x-axis, in 
%        anti-clockwise manner.
%     2. C, a 2 x 2 covariance matrix, the ellipse is the set of points
%        with mahalanobis distance to the center being equal to 1.
%        ellipse eqn: x^T C x = 1
%     3. {C, r}, C is a 2 x 2 covariance matrix, the ellipse is the set 
%        of points with mahalanobis distance to the center being equal to 
%        r. ellipse eqn: x^T C x = r^2.
%
%   - sldrawellipse(center, shape, n) draws the ellipse using specified
%     number of samples. If n is not specified or empty, then n takes
%     the default value 300.
%          
%   - sldrawellipse(center, shape, n, ...) draws the ellipse with the 
%     plotting properties specified by additional arguments. By default,
%     it uses 'b-'.
%
%   - h = sldrawellipse(...) returns the handle to the line
%     drawn on the axis.
%
% $ History $
%   - Created by Dahua Lin on Apr 23, 2006
%   - Modified by Dahua Lin, on Aug 26, 2006
%       - from sldraw_ellipse to sldrawellipse
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%


%% parse and verify arguments

if nargin < 2
    raise_lackinput('sldraw_ellipse', 2);
end

if length(center) ~= 2
    error('sltoolbox:invaliddims', 'center should be a vector of length 2');    
end
center = center(:);

if nargin < 3 || isempty(n)
    n = 300;
end

%% convert the shape to transform (scale and rotation)

if isnumeric(shape) && length(shape) == 3 % [a, b, theta] form    
    
    [S, R] = get_transform_from_geo(shape(1), shape(2), shape(3));                
    
elseif isnumeric(shape) && isequal(size(shape), [2, 2]) % C form
    
    [S, R] = get_transform_from_cov(shape);
    
elseif iscell(shape) && length(shape) == 2 % {C, r} form
    
    [S, R] = get_transform_from_cov(shape{1});
    S = S * shape{2};
    
end

T = R * S;

%% make the points forming the ellipse

% make a circle first
t = linspace(0, 2*pi, n);
X = [cos(t); sin(t)];

% transform the circle to an ellipse
X = T * X;
X = sladdvec(X, center, 1);

% draw the circle
hold on;
h = plot(X(1,:), X(2,:), varargin{:});


%% The functions for making transforms from shapes

function [S, R] = get_transform_from_geo(a, b, theta)

S = [a 0; 0 b];
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

function [S, R] = get_transform_from_cov(C)

[S, R] = slsymeig(C);
S = max(S, 0);  % a zero guard
S = diag(sqrt(S));



    
    


