function [A, b] = sllinrega(X, Y, varargin)
%SLLINREGA Performs Augmented Multivariate Linear Regression
%
% $ Syntax $
%   - [A, b] = sllinrega(X, Y, ...)
%
% $ Arguments $
%   - X:        The sample matrix of x
%   - Y:        The sample matrix of y
%   - A:        The solved transform matrix
%   - b:        The solved shift vector
%
% $ Description $
%   - [A, b] = sllinrega(X, Y, ...) solves the regression problem given
%     by the following formula:
%           y = A * x + b
%     in least square error sense. The samples are stored in X and Y
%     in column-wise manner.
%     You can specify properties for regression as in sllinreg.
%
% $ Remarks $
%   - The implementation is based on sllinreg with an augmented 
%     formulation as follows:
%           y = [A, b] * [x; 1]
%
% $ History $
%   - Created by Dahua Lin, on Sep 15th, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sllinrega', 2);
end

if ~isnumeric(X) || ~isnumeric(Y) || ndims(X) ~= 2 || ndims(Y) ~= 2
    error('sltoolbox:invalidarg', ...
        'The X and Y should be both 2D numeric matrices');
end

%% main

% augment formulation
[dx, nx] = size(X);
Xa = [X; ones(1, nx)];

% solve
Aa = sllinreg(Xa, Y, varargin{:});
clear Xa;

% extract 
A = Aa(:, 1:dx);
b = Aa(:, dx+1);







