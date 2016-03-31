function [evals, evecs] = slsymgeig(A, B, method, r)
%SLSYMGEIG Solve the generalized eigen decomposition for symmetric matrices
%
% $ Syntax $
%   - [evals, evecs] = slsymgeig(A, B)
%   - [evals, evecs] = slsymgeig(A, B, method, r)
%
% $ Description $
%   - [evals, evecs] = slsymgeig(A, B) solves the generalized eigenvalue
%     and eigenvectors problem by default method. The problem is formulated
%     by A * v = lambda * B * v. The default method is 'std'.
%
%   - [evals, evecs] = slsymeig(A, B, method, r) solves the generalized
%     eigenvalue and eigenvectors problem using the specified method.
%     r is the parameter for the method.
%     \*
%     \t   Table 1. Methods for Solving symmetric GEVD problem  \\
%           name    &   description                             \\
%          'std'    &  Keep only the eigenvectors of B, which correspond
%                      to non-trivial eigenvalues. r is the ratio of
%                      minimum allowable nontrivial eigenvalues to the
%                      maximum eigenvalue.  (default r = 1e-7) \\
%          'reg'    &  Regularize the eigenvalues by adding a small 
%                      positive value to all eigenvalues. r is the ratio
%                      of the addend to the maximum eigenvalue. 
%                      (default r = 1e-6)                      \\
%          'bound'  &  Enforce a lower bound on the eigenvalues. All 
%                      eigenvalues smaller than the lower bound is 
%                      forced to be the lower bound. r is the ratio of
%                      the lower bound to the maximum eigenvalue. 
%                      (default r = 1e-6)                       \\
%     \*
%
% $ Remarks $
%   -# It is required that B be a positive semidefinite matrix.
%
%   -# In the output, the eigenvalues are sorted in descending order,
%      with the eigenvectors arranged according to the eigenvalues.
%      
%   -# The eigenvectors are scaled, so that v^T * B * v = 1. Moreover,
%      for the whole set of eigenvectors V, it has V^T * B * V = I.
%
%   -# The problem are solved as follows: first solve the whiten transform
%      B^(-1/2), then solve the standard eigen-problem given by the matrix
%      B^(-1/2) * A * B^(-1/2) and obtain the eigenvectors Y. Finally, we
%      compute the generalized eigenvectors by B^(-1/2) * Y. It can be
%      shown that the generalized eigenvalues are equal to those of the
%      converted standard eigen-problem.
%
%   -# In mathematically, solve the generalized eigen-problem can be
%      considered as solving the following optimization problem, given
%      that B is non-singular:
%           max v^T * A * v,  s.t. v^T * B * v = I.
%
% $ History $
%   - Created by Dahua Lin on May 3rd, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - replace slmul by slmulvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slsymgeig', 2);
end

% for A and B
d = size(A, 1);
if ~isequal(size(A), [d d])
    error('sltoolbox:invaliddims', ...
        'A should be a sqaure matrix');
end
if ~isequal(size(B), [d d])
    error('sltoolbox:sizmismatch', ...
        'The size of B is inconsistent with that of A');
end

% for method
if nargin < 3 || isempty(method)
    method = 'std';
end

% for r
if nargin < 4 || isempty(r)
    use_default_r = true;
else
    use_default_r = false;
end


%%  Configure methods

switch method
    case 'std'
        fh_proc_evals = @proc_evals_std;
        if use_default_r
            r = 1e-7;
        end
    case 'reg'
        fh_proc_evals = @proc_evals_reg;
        if use_default_r
            r = 1e-6;
        end
    case 'bound'
        fh_proc_evals = @proc_evals_bound;
        if use_default_r
            r = 1e-6;
        end
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid method %s for slsymgeig', method);
end

%% Compute

% step 0: enforce symmetry for A and B
A = (A + A') / 2;
B = (B + B') / 2;


% step 1: solve the whiten transform B^(1/2)

[B_evals, B_evecs] = slsymeig(B);
B_inv_evals = fh_proc_evals(B_evals, r);
clear B_evals;
B_inv_evals = max(B_inv_evals, 0);    % enforce non-negative
k = length(B_inv_evals);
if k < d
    B_evecs = B_evecs(:, 1:k);
end
W = slmulvec(B_evecs, sqrt(B_inv_evals)');
clear B_evecs B_inv_evals;

% step 2: solve the converted EVD problem

A = W' * A * W;
[evals, Y] = slsymeig(A);

% step 3: convert the eigenvectors

evecs = W * Y;


%% The sub-functions for processing eigenvalues

function revs = proc_evals_std(evs, r)

lb = r * evs(1);
k = sum(evs > lb);
revs = 1 ./ evs(1:k);

function revs = proc_evals_reg(evs, r)

a = r * evs(1);
revs = 1 ./ (max(evs, 0) + a);

function revs = proc_evals_bound(evs, r)

lb = r * evs(1);
k = sum(evs > lb);
if k < length(evs)
    evs(k+1:end) = lb;
end
revs = 1 ./ evs;
