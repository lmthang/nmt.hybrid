function X = slpcarecon(S, Y)
%SLPCARECON Reconstructs the samples in original space
%
% $ Syntax $
%   - Xr = slpcarecon(S, Y)
%
% $ Arguments $
%   - S:        the PCA model struct
%   - Y:        the principal component features
%   - Xr:       the reconstructed samples
%
% $ Description $
%   - Xr = slpcarecon(S, Y) reconstructs the original samples approximately
%     using the principal components Y. If the dimension of Y is less than
%     the subspace dimension, the leading space dimensions will be used.
%
% $ History $
%   - Created by Dahua Lin, on Aug 17, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace sladd by sladdvec to increase efficiency
%

%% parse and verify input

if ~isstruct(S)
    error('sltoolbox:invalidarg', ...
        'S should be a PCA model struct');
end

if ~isnumeric(Y) || ndims(Y) ~= 2
    error('sltoolbox:invalidarg', ...
        'The features Y should be a 2D numeric matrix');
end

dy = size(Y, 1);
if dy > S.feadim
    error('sltoolbox:sizmismatch', ...
        'The feature dimension of Y exceeds the subspace dimension preserved in model');
end

%% reconstruct

if dy == S.feadim
    X = S.P * Y;
else
    X = S.P(:, 1:dy) * Y;
end

X = sladdvec(X, S.vmean, 1);




