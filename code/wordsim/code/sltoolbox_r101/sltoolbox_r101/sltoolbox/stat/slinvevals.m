function revs = slinvevals(evals, method, r)
%SLINVEVALS Compute the reciprocals of eigenvalues in a robust way
%
% $ Syntax $
%   - revs = slinvevals(evals)
%   - revs = slinvevals(evals, method, r)
%
% $ Description $
%   - revs = slinvevals(evals) computes the reciprocals of eigenvalues in
%     the default way: using the default method and its corresponding
%     default r value.
%   
%   - revs = slinvevals(evals, method, r) computes the reciprocals of 
%     eigenvalues in a user-specified way.
%     \*
%     \t   Table 1. The methods of computing eigenvalue reciprocals
%          name      &        revs
%          'std'     & For effective eigenvalues, their reciprocals are
%                      computed as usual; for the rest ones, their 
%                      reciprocals are set to zeros. r values here is 
%                      the ratio of minimum allowable effective eigenvalues
%                      to the maximum eigenvalue. default r = 1e-7;
%          'reg'     & Regularize the eigenvalues before computing their
%                      reciprocals. By regularization, a small positive 
%                      value is added to all eigenvalues. r value here is
%                      the ratio of the addend to the maximum eigenvalue.
%                      default r = 1e-6.
%          'bound'   & Enforce lower bound to eigenvalues before computing
%                      their reciprocals. The eigenvalues below the lower
%                      bound is set to the lower bound value. r value here
%                      is the ratio of the lower bound to the maximum 
%                      eigenvalue. default r = 1e-6.
%          'gapprox' & Computing the reciprocals of the eigenvalues in the
%                      way of optimal non-singular approximation of 
%                      Gaussian distribution. r values here is the ratio
%                      of the minimum effective eigenvalues to the
%                      maximum eigenvalue. The eigenvalues below are 
%                      considered to be corresponding to isometric noises.
%                      default r = 1e-6.
%     \*
%
% $ Remarks $
%   - The eigenvalues should be a column vector with values arranged in 
%     a descending order.
%
% $ History $
%   - Created by Dahua Lin on Apr 30th, 2006
%

%% parse and verify input arguments

n = length(evals);
if n ~= numel(evals)
    error('sltoolbox:invalidarg', 'evals should be a vector');
end

if nargin < 2 || isempty(method)
    method = 'std';
end

%% compute

% make strictly non-negative
evals = max(evals, 0);

switch method    
    case 'std'
        if nargin < 3 || isempty(r)
            r = 1e-7;
        end
        lb = r * evals(1);
        k = sum(evals >= lb);
        if k == n
            revs = 1 ./ evals;
        else
            revs = [1 ./ evals(1:k); zeros(n-k, 1)];
        end
        
    case 'reg'
        if nargin < 3 || isempty(r)
            r = 1e-6;
        end
        a = r * evals(1);
        revs = 1 ./ (evals + a);
        
    case 'bound'
        if nargin < 3 || isempty(r)
            r = 1e-6;
        end
        lb = r * evals(1);
        evals = max(evals, lb);
        revs = 1 ./ evals;
        
    case 'gapprox'
        if nargin < 3 || isempty(r)
            r = 1e-6;
        end
        lb = r * evals(1);
        k = sum(evals >= lb);
        if k == n
            revs = 1 ./ evals;
        else
            nv = sum(evals(k+1:n)) / (n-k);
            rnv = 1 / nv;
            rnvs = rnv(ones(n-k, 1));
            revs = [1 ./ evals(1:k); rnvs];
        end
        
end
            
