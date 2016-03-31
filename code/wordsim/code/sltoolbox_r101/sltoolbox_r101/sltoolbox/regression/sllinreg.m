function A = sllinreg(X, Y, varargin)
%SLLINREG Performs Multivariate Linear Regression and Ridge Regression
%
% $ Syntax $
%   - A = sllinreg(X, Y, ...)
%
% $ Arguments $
%   - X:        The matrix of x samples
%   - Y:        The matrix of y samples
%   - A:        The solved transform matrix
%
% $ Description $
%   - A = sllinreg(X, Y, varargin) solves the linear regression problem to
%     get the linear transforms A: (y = Ax + e). The solution is given by 
%     the following optimization problem:
%       A = argmin_{A} sum_i ||y_i - A x_i||^2 + sum_k lambda_k ||a_k||^2
%     Here, Y is a dy x n matrix with the i-th column giving y_i; 
%           X is a dx x n matrix with the i-th column giving x_i
%           A is a dy x dx transform matrix
%           a_k is the k-row vector of A, which corresponds to the k-th 
%                  component of y
%           lambda_k is the regularization weight in ridge regression
%     According to mathematical analysis, the solution is given as
%       A = (Y * X^T) * (X * X^T + diag(lambda))^{-1}
%     You can specify the following properties to control the regression:
%     \*
%     \t    Table  Linear Regression Properties
%     \h        name       &             description
%              'lambdas'   &  The regularization coefficients in ridge 
%                             linear regression. If all components share
%                             the same value, then lambdas can be a scalar.
%                             If different values are for different 
%                             dimensions, then lambdas can be a dy x 1 
%                             column vector. (default = 0)
%              'weights'   &  The sample weights, can be either [] or
%                             an 1 x n row vector.
%              'invparams' &  The parameters for invoking slinvcov to 
%                             compute inverse matrix, in the form of
%                             {method, ...}. default = [], means directly
%                             using inv to do inverse.
%     \*
%
% $ Remarks $
%   - To solve the linear problem like y = Ax + b, you have two ways:
%     (1) centralize x and y respectively, and invoke sllinreg, then set
%         b = my - A * mx
%     (2) invoke sllinrega, which is specially designed for such an
%         augmented problem.
%
% $ History $
%   - Created by Dahua Lin, on Sep 15, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sllinreg', 2);
end

if ~isnumeric(X) || ~isnumeric(Y) || ndims(X) ~= 2 || ndims(Y) ~= 2
    error('sltoolbox:invalidarg', ...
        'The X and Y should be both 2D numeric matrices');
end

[dx, n] = size(X);
[dy, ny] = size(Y);
if n ~= ny
    error('sltoolbox:sizmismatch', ...
        'X and Y contain different numbers of samples');
end

opts.lambdas = 0;
opts.rv = 1e-3;
opts.weights = [];
opts.invparams = [];
opts = slparseprops(opts, varargin{:});

lambdas = opts.lambdas;
if ~isscalar(lambdas) && ~isequal(size(lambdas), [dy, 1])
    error('lambdas should be either a scalar or a dy x 1 column vector');
end

w = opts.weights;
if ~isempty(w)
    if ~isequal(size(w), [1, n])
        error('The sample weights should be a 1 x n row vector');
    end
end


%% main

if isempty(w)
    Xt = X';
else
    Xt = slmulvec(X, w, 2)';
end

M2 = X * Xt;    
if ~isequal(lambdas, 0)
    inds = (1:dx)' * (dx+1) - dx;
    M2(inds) = M2(inds) + lambdas;
end

if isempty(opts.invparams)
    invM2 = inv(M2);
else
    invM2 = slinvcov(M2, opts.invparams{:});
end
clear M2;

M1 = Y * Xt;

A = M1 * invM2;

    
    











