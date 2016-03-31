function [A, b, props, info] = sllogistreg(X, nums, varargin)
%SLLOGISTREG Performs Multivariate Logistic Regression
%
% $ Syntax $
%   - [A, b] = sllogistreg(X, nums, ...)
%   - [A, b, props] = sllogistreg(X, nums, ...)
%   - [A, b, props, info] = sllogistreg(X, nums, ...)
%
% $ Arguments $
%   - X:            The input sample matrix
%   - nums:         The numbers of samples in different classes
%   - A:            The solved logistic coefficients
%   - b:            The solved logistic shift value
%   - props:        The probability values under the resultant model
%   - info:         The information of learning process
%                   - exitflag: The exitflag given by fminunc
%                   - numiters: The number of iterations
%                   - fval:     The final objective value
%
% $ Description $
%   - [A, b] = sllogistreg(X, nums, ...) solves the multivariate logistic 
%     regression. Suppose there are n samples, from C classes, then C
%     models are learned for the C classes. The k-th model is characterized
%     by a coefficient vector a_k and a shift value b_k. Then a sample x,
%     the probability that it belongs to the k-th class is given by the
%     following formula:
%           p(x, k) = P_k * h_k(x) / sum_l (P_l * h_l(x))
%     here, we have
%           h_k(x) = a_k^T * x + b_k
%     The objective of logistic regression is to maximize the sum of 
%     the probabilities that the samples belong to its own class:
%           maximize sum_i p(x_i, c_i)
%     here, c_i is the class label corresponding to x_i.
%     In the input arguments, X should be a d x n matrix containing the
%     x samples with the samples from the same class gathering together,
%     nums should be a 1 x C row vector. 
%     In the output arguments, A should be a d x C matrix, and b is a
%     1 x C row vector. Each column in A together with the corresponding
%     shift value in b describe a model for one class.
%
%     You can specify the following parameter to control the numeric 
%     optimization process:
%       - 'weights':        The weights of the samples  (1 x n)
%                           default = []: means all weights are 1
%       - 'priors':         The priors of the classes (1 x C)
%                           default = []: means equal priors
%       - 'maxiter':        The maximum number of iterations
%                           default = 300
%       - 'tolF':           The tolerance of the change of objective func
%                           default = 1e-6
%       - 'tolX':           The tolerance of the change of parameters
%                           default = 1e-6
%       - 'display':        The level of display
%                           default = 'off'
%       - 'init':           A cell array as {A0, b0}
%                           default = {}, using random initialization
%
%   - [A, b, props] = sllogistreg(X, nums, ...) also output the 
%     probability values that the input samples belong to the true class.
%     It would be a 1 x n row vector.
%
%   - [A, b, props, info] = sllogistreg(X, nums, ...) outputs the info
%     about the optimization process
% 
% $ Remarks $
%   - This function is implemented based on the optimization function
%     fminunc and fmincon, and thus requires the optimization toolbox 
%     be installed.
%
% $ History $
%   - Created by Dahua Lin, on Sep 16, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('sllogistreg', 2);
end

[d, n] = size(X);
if ~isvector(nums) || size(nums, 1) ~= 1
    error('sltoolbox:invalidarg', ...
        'nums should be a 1 x C row vector');
end
if sum(nums) ~= n
    error('sltoolbox:sizmismatch', ...
        'The numbers in nums are not consistent with the sample number');
end

C = length(nums);

if C < 2
    error('sltoolbox:invalidarg', ...
        'There should be at least 2 classes.');
end

opts.weights = [];
opts.priors = [];
opts.maxiter = 300;
opts.tolF = 1e-6;
opts.tolX = 1e-6;
opts.display = 'off';
opts.init = {};
opts = slparseprops(opts, varargin{:});

w = opts.weights;
if ~isempty(w)
    if ~isequal(size(w), [1, n])
        error('sltoolbox:sizmismatch', ...
            'w should be a 1 x n row vector');
    end
end

pri = opts.priors;
if ~isempty(pri)
    if ~isequal(size(pri), [1, C])
        error('sltoolbox:sizmismatch', ...
            'pri should be a 1 x C row vector');
    end
end

if isempty(opts.init)
    is_inited = false;
else
    A0 = opts.init{1};
    b0 = opts.init{2};
    if ~isequal(size(A0), [d, C])
        error('sltoolbox:sizmismatch', 'A0 should be a d x C matrix');
    end
    if ~isequal(size(b0), [1, C])
        error('sltoolbox:sizmismatch', 'b0 should be a 1 x C row vector');
    end
    is_inited = true;    
end


%% main

%NOTE: we convert the problem of maximization to minimization of the
%      negative object

% augment problem

Xa = [X; ones(1, n)];

% prepare for optimization
optimfunc = @(v) logistic_objfun(v, Xa, nums, d, n, C, pri, w);
optimopts = optimset(optimset('fminunc'), ...
    'LargeScale', 'on', ...
    'GradObj', 'on', ...
    'MaxIter', opts.maxiter, ...
    'Display', opts.display, ...
    'TolFun', opts.tolF, ...
    'TolX', opts.tolX);
    
% initialization
if ~is_inited
    v0 = rand((d+1)*C, 1);
else
    v0 = reshape([A0; b0], (d+1)*C, 1);
end

% do optimization
[v, fval, exitflag, optimoutput] = fminunc(optimfunc, v0, optimopts);
clear v0;
clear optimfunc;
clear Xa;

% make output
v = reshape(v, d+1, C);
A = v(1:d, :);
b = v(d+1, :);
clear v;

if nargout >= 3
    L = compute_logit(A, X) ;
    L = sladdvec(L, b', 1);
    props = slposterioritrue(L, nums, pri, 'log');            
end

if nargout >= 4
    info.exitflag = exitflag;
    info.numiters = optimoutput.iterations;
    info.fval = -fval; 
end


%% The core functions for objective and gradient evaluation

function [f, g] = logistic_objfun(v, Xa, nums, d, n, C, pri, w)
% v is a reshape version of Aa

% get input
Aa = reshape(v, d+1, C);
L = compute_logit(Aa, Xa);
clear Aa;

% compute f
P = slposteriori(L, pri, 'log');
[sps, eps] = slnums2bounds(nums);
pps = zeros(1, n);
for k = 1 : C
    sk = sps(k); ek = eps(k);
    pps(sk:ek) = P(k, sk:ek);
end

if isempty(w)
    f = -sum(log(pps));
else
    f = -sum(log(pps) .* w);
end

% compute g
M = make_indicatormap(nums, C, n);
M = M - P;
clear P;
if ~isempty(w)
    M = slmulvec(M, w, 2);
end
g = Xa * M';
clear M;
g = -g(:);


function L = compute_logit(Aa, Xa)
% L is the logit values (a_k' * x_i + b_k): C x n matrix
L = Aa' * Xa;


%% Auxiliary functions

function M = make_indicatormap(nums, C, n)
% C x n
% C(i, j) = 1, when sample j belongs to class i

M = zeros(C, n);
I = slexpand(nums);
J = 1:n;
inds = sub2ind([C, n], I, J);
clear I J;
M(inds) = 1;

