function K = slkernel(varargin)
%SLKERNEL Computes the kernel for samples
%
% $ Syntax $
%   - K = slkernel(X0, kernel_type, ...)
%   - K = slkernel(X0, X, kernel_type, ...)
%
% $ Description $
%   - K = slkernel(X0, kernel_type, ...) Computes the Gram matrix for
%     the samples in matrix X0 using the kernel specified by kernel_type. 
%     kernel_type can be name of a built-in kernel type, or the name, 
%     handle of an user-specified function. The user-specified function
%     should take in two d x n matrix containing n pairs of vectors, and
%     output a 1 x n vector of their kernel values. The available builtin
%     kernels are given as follows:
%     \*
%     \t   Table 1.  Built-in Kernel Types          \\
%     \h     name     &    description              \\
%           'lin'     &  Linear Kernel: 
%                        x1' * x2, 
%                        with no parameters         \\
%           'gauss'   &  Gaussian RBF Kernel:
%                        exp(- ||x1 - x2||^2 / (2 * sigma^2)), 
%                        with one parameter sigma   \\ 
%           'poly'    &  Polynomial Kernel:
%                        ((x1' * x2) + a)^k,
%                        with two parameters: k and a \\
%           'sigmoid' &  Sigmoidal Kernel:
%                        tanh(k * (x1' * x2) + a),    
%                        with two parameters: k and a \\
%           'invquad' &  Inverse Quadratic Kernel:
%                        1 / sqrt(||x1 - x2||^2 + a), 
%                        with one parameter: a        \\
%     \* 
%
%   - K = slkernel(X0, X, kernel_type, p1, p2, ...) Computes the empirical
%     kernel mapping for samples X with respect to data set X0. Suppose
%     there are n0 samples in X0, n samples in X, then K will be an n0 * n
%     matrix. Each column in K corresponds to a column in X.
%
% $ History $
%   - Created by Dahua Lin on Jul 13rd, 2005
%   - Modified by Dahua Lin on May 2nd, 2005
%       - base on sltoolbox v4
%       - re-organize the code structure
%       - add the ability of user-specified kernel functions.
%

%% parse and verify input arguments

% for X0
if ~isnumeric(varargin{1})
    error('sltoolbox:invalidarg', ...
        'The first argument should be an numeric matrix');
end
X0 = varargin{1};
if ndims(X0) ~= 2
    error('sltoolbox:invaliddims', ...
        'The X0 should be a 2D matrix');
end

% for X
if isnumeric(varargin{2})
    X = varargin{2};    
    if ndims(X) ~= 2
        error('sltoolbox:invaliddims', ...
            'X should be a 2D matrix');
    end
    ipkt = 3;       % argument index of kernel type    
else
    X = X0;
    ipkt = 2;        
end

% for kernel type
if nargin < ipkt || isempty(varargin{ipkt})
    error('sltoolbox:invalidarg', ...
        'kernel type is not specified');
end
kernel_type = varargin{ipkt};

% for extra parameters
if nargin == ipkt       % no extra parameters
    params = {};
else
    params = varargin(ipkt+1:end);
end


%% Determine the computation routine and delegate to it

% determine for the built-in ones

bik = false;
if ischar(kernel_type)
    switch kernel_type
        case 'lin'
            bik = true;   % bik means built-in kernel
            fh_kernel = @lin_kernel;
        case 'gauss'
            bik = true;
            fh_kernel = @gauss_kernel;
        case 'poly'
            bik = true;
            fh_kernel = @poly_kernel;
        case 'sigmoid'
            bik = true;
            fh_kernel = @sigmoid_kernel;
        case 'invquad'
            bik = true;
            fh_kernel = @invquad_kernel;
    end
end

% delegate
if bik
    K = fh_kernel(X0, X, params{:});
else
    K = slpweval(X0, X, kernel_type, params{:});
end


%% The built-in kernel functions

% Linear Kernel
function K = lin_kernel(X0, X)

K = X0' * X;


% Gaussian Radius Base Function Kernel
function K = gauss_kernel(X0, X, sigma)

D2 = slmetric_pw(X0, X, 'sqdist');
K = exp(- D2 / (2 * sigma * sigma));


% Polynomial Kernel
function K = poly_kernel(X0, X, k, a)

K = (X0' * X + a).^k;


% Sigmoidal Kernel
function K = sigmoid_kernel(X0, X, k, a)

K = tanh(k * (X0' * X) + a);


% Inverse Quadratic Kernel
function K = invquad_kernel(X0, X, a)

D2 = slmetric_pw(X0, X, 'sqdist');
K = 1 ./ sqrt(D2 + a);


