function dists = slgaussmdist(GS, X)
%SLGAUSSMDIST Computes the Malanobis distance between samples and centers
%
% $ Syntax $
%   - dists = slgaussmdist(GS, X)
%
% $ Arguments $
%   - GS:       the Gaussian models
%   - X:        the sample matrix
%   - dists:    the distances of samples to the model centers
%       
% $ Description $
%   - dists = slgaussmdist(GS, X) computes the distances from the samples
%     in X and the model centers of GS. If there are n samples in X, and
%     k models in GS, then dists is a k x n matrix. Each column of dists
%     is the distances from the corresponding sample to all model centers.
%     The distances for each model is computed based on the variances or
%     covariances of that model.
% 
% $ History $
%   - Created by Dahua Lin, on Aug 28, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slgaussmdist', 2);
end

if ~isnumeric(X) || ndims(X) ~= 2
    error('sltoolbox:invalidarg', ...
        'The X should be a 2D numeric matrix');
end

tyi = slgausstype(GS);

if ~tyi.hasinv
    error('sltoolbox:invalidarg', ...
        'GS should have inverse variance/covariance computed');
end

[d, n] = size(X);
if d ~= GS.dim
    error('sltoolbox:sizmismatch', ...
        'The dimension of samples does not match that of the models');
end

k = GS.nmodels;

%% Main skeleton

switch tyi.varform
    case 'univar'
        if tyi.sharevar
            dists = compmdist_univar(GS.means, X, GS.invvars);
        else
            dists = zeros(k, n);
            for i = 1 : k
                dists(i, :) = compmdist_univar(GS.means(:,i), X, GS.invvars(i));
            end
        end
    case 'diagvar'
        if tyi.sharevar
            dists = compmdist_diagvar(GS.means, X, GS.invvars);
        else
            dists = zeros(k, n);
            for i = 1 : k
                dists(i, :) = compmdist_diagvar(GS.means(:,i), X, GS.invvars(:,i));
            end
        end
    case 'covar'
        if tyi.sharevar
            dists = compmdist_covar(GS.means, X, GS.invcovs);
        else
            dists = zeros(k, n);
            for i = 1 : k
                dists(i, :) = compmdist_covar(GS.means(:,i), X, GS.invcovs(:,:,i));
            end
        end
end


%% Core computation routines

function dists = compmdist_univar(M, X, invvar)

dists = slmetric_pw(M, X, 'sqdist');
dists = dists * invvar;
dists = sqrt(max(dists, 0));

function dists = compmdist_diagvar(M, X, invvars)

dists = slmetric_pw(M, X, 'wsqdist', invvars);
dists = sqrt(max(dists, 0));

function dists = compmdist_covar(M, X, invcovs)

dists = slmetric_pw(M, X, 'quaddiff', invcovs);
dists = sqrt(max(dists, 0));






