function C = slpoolcov(Cs, w)
%SLPOOLCOV Compute the pooled covariance 
%
% $ Syntax $
%   - C = slpoolcov(Cs)
%   - C = slpoolcov(Cs, w)
%
% $ Arguments $
%   - Cs        the stack of covariance matrices for all components
%   - w:        the weights for components
%   - C:        the pooled covariance matrix
%
% $ Description $
%   - C = slpoolcov(Cs) computes the average covariance matrix.
%
%   - C = slpoolcov(Cs, w) computes the weighted pooled covariance matrix
%     with the component weights specified in w.
%
% $ History $
%   - Created by Dahua Lin on Dec 19th, 2004
%   - Modified by Dahua Lin on Apr 22nd, 2004
%       - Give an much more efficient implementation using matrix product
%         and reshaping.
%

%% parse and verify input arguments
[d1, d2, n] = size(Cs);
if d1 ~= d2
    error('sltoolbox:notsquaremat', ...
        'the covariance matrices should be square');
end
d = d1;
if nargin < 2 || isempty(w)
    isweighted = false;
else
    isweighted = true;
    if numel(w) ~= n
        error('sltoolbox:argmismatch', ...
            'the size of w does not match that of Cs');
    end
end

%% compute
if n == 1
    C = Cs;
else
    if ~isweighted
        Cs = reshape(Cs, [d*d, n]);
        C = reshape(sum(Cs, 2), [d, d]);
    else        
        w = w(:) / sum(w(:));
        Cs = reshape(Cs, [d*d, n]);
        C = reshape(Cs * w, [d, d]);
    end
end





    



