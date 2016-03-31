function S = slkernelscatter(K, type, varargin)
%SLKERNELSCATTER Compute the kernelized scatter matrix
%
% $ Syntax $
%   - S = slkernelscatter(K, type, ...)
%
% $ Arguments $
%   - K:        the kernel gram matrix of the samples
%   - type:     the type of the scatter matrix
%   - S:        the resulting scatter matrix
%
% $ Description $
%   - S = slkernelscatter(K, type, ...) computes the kernelized scatter
%     matrix of K. It can be shown that the computation of the kernelized
%     scatter matrix is equivalent to the computation of conventional 
%     scatter matrix with the sample matrix replaced by the gram matrix.
%     Thus this function simply invoke slscatter with K replacing X.
%     The usage can be referred to function slscatter.
%
% $ Remarks $
%   -# The so-called kernel scatter matrix is an n x n matrix defined by
%      following formula:
%       S = Phi^T * scatter(phi_1, phi_2, ..., phi_n) * Phi
%      here scatter(.) is the scatter matrix defined like for conventional
%      scatter but on the nonlinearly mapped features. Phi is the set of
%      nonlinearly mapped features. The kernelized scatter matrix plays 
%      a core role in kernelized discrminant analysis.
%
% $ History $
%   - Created by Dahua Lin on May 03, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slkernelscatter', 2);
end

if ndims(K) ~= 2 || size(K, 1) ~= size(K, 2)
    error('sltoolbox:invaliddims', ...
        'The gram matrix K should be a square matrix');
end

%% delegate to slscatter for computation

S = slscatter(K, type, varargin{:});
