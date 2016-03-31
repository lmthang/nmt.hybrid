function Crs = slregcov(Cs, varargin)
%SLREGCOV Regularizes covariance matrices
%
% $ Syntax $
%   - Crs = slregcov(Cs, ...)
%
% $ Description $
%   - Crs = slregcov(Cs, ...) regularizes the covariance matrix in Cs
%   the output the regularized covariance matrix by Crs. You can specify
%   additional properties to control the process of regularization
%
%   The regularization follows the following formula
%              (1 - lambda) * P * C + lambda * Cpool
%        C1 = ------------------------------------------
%               (1 - lambda) * P + lambda
%
%        Cr = (1 - gamma) * C1 + gamma * (tr(C1)/d) * I 
%
%   \t  Table:  Properties of Covariance Regularization
%   \h  property name  &  property description
%       'lambda'          the lambda coefficient (0 <= lambda <= 1), default = 0
%       'gamma'           the gamma coefficient (0 <= gamma <= 1), default = 0
%       'prior'           the priori (by default they are set to equal)
%       'poolcov'         the pool covariance matrix
%       
% $ Remarks $
%   - If poolcov is not given, it will be computed by averaging the
%     covariance matrix, weighted by prior.
% 
% $ History $
%   - Created by Dahua Lin on Dec 19th, 2005
%

%% verify and parse input arguments
[d1, d2, k] = size(Cs);
if d1 ~= d2
    error('sltoolbox:notsquaremat', ...
        'The covariance matrices should be square');
end
d = d1;
S.lambda = 0;
S.gamma = 0;
S.prior = [];
S.poolcov = [];
S = slparseprops(S, varargin{:});
if isempty(S.prior)
    ispriorgiven = false;
    S.prior = ones(1, k) / k;
else
    ispriorgiven = true;
    S.prior = S.prior(:)';
    if length(S.prior) ~= k
        error('sltoolbox:sizmismatch', ...
            'The length of prior is not consistent with the number of covariances');
    end
    S.prior = S.prior / sum(S.prior);
end
if S.lambda < 0 || S.lambda > 1
    error('sltoolbox:invalidarg', ...
        'lambda should be within [0, 1]');
end
if S.gamma < 0 || S.gamma > 1
    error('sltoolbox:invalidarg', ...
        'gamma should be within [0, 1]');
end
if ~isempty(S.poolcov)
    ispoolgiven = true;
    if ~isequal(size(S.poolcov), [d d]);
        error('sltoolbox:invalidarg', ...
            'The size of pooled covariance is illegal');
    end
else
    ispoolgiven = false;
end


%% get the pool covariance
if ~ispoolgiven
    if ~ispriorgiven
        S.poolcov = slpoolcov(Cs);
    else
        S.poolcov = slpoolcov(Cs, S.prior);
    end
end


%% lambda-regularize
lambda = S.lambda;
if lambda == 0
    Crs = Cs;
elseif lambda == 1
    Crs = zeros(size(Cs));
    for i = 1 : k
        Crs(:, :, i) = S.poolcov;
    end
else
    Crs = zeros(size(Cs));
    coeffsL = zeros(2, k);
    coeffsL(1, :) = (1 - lambda) * S.prior;
    coeffsL(2, :) = lambda;
    coeffsL = slnormalize(coeffsL, 1, 1);
    for i = 1 : k
        Crs(:,:,i) = coeffsL(1, i) * Cs(:,:,i) + coeffsL(2, i) * S.poolcov;
    end
end


%% gamma-regularize
gamma = S.gamma;
if gamma == 0
    % nothing to do
elseif gamma == 1
    for i = 1 : k 
        avgtr = trace(Crs(:,:,i)) / d;
        Crs(:,:,i) = avgtr * eye(d, d);
    end
else
    for i = 1 : k
        C0 = Crs(:,:,i);
        avgtr = trace(C0) / d;
        Crs(:,:,i) = (1-gamma) * C0 + gamma * avgtr * eye(d, d);
    end
end

     
