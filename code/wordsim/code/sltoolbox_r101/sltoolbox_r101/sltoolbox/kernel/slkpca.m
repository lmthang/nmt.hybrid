function [A, evs] = slkpca(K0, varargin)
%SLPCA Learns a Kernel PCA model from training samples
%
% $ Syntax $
%   - A = slkpca(K0)
%   - A = slkpca(K0, ...)
%   - [A, evs] = slkpca(K0, ...)
%
% $ Arguments $
%   - X:        the training sample matrix
%   - A:        the projection coefficient matrix 
%   - evs:      the egienvalues of all n0-1 dimensions 
%               (including preserved and discarded)
%
% $ Description $
%   - A = slpca(K0) learns a Kernel PCA model from the samples X by 
%     default settings.
%
%   - A = slpca(K0, ...) learns a Kernel PCA model from the samples X 
%     according to the properties specified:
%     \*
%     \t   Table 1. The properties of Kernel PCA learning
%     \h    name       &    description                               \\
%          'preserve'  & Determine how many components are preserved, it is
%                        given in following form: {sch, ...}, which is used
%                        as parameters in sldim_by_eigval.       \\
%          'weights'   & The 1 x n row vector of sample weights.  
%                        If the weights are not specified, then it 
%                        considers that all samples have equal weights. 
%                        default = [].   \\
%        'centralized' & Whether the K0 has been centralized. default =
%                        false. If the K0 is not centralized, it will 
%                        first centralize it before subsequent steps. \\
%     \*
%  
% $ Remarks $
%   -# The coefficient vectors are normalized so that a^T * K * a = 1.
%
% $ History $
%   - Created by Dahua Lin on May 02, 2006
%   - Modified by Dahua Lin on Sep 10, 2006
%       - use slmulvec and slmulrowcols to increase efficiency
%         in the weighted cases
%

%% parse and verify input arguments

% for K0
n0 = size(K0, 1);
if ~isequal(size(K0), [n0 n0])
    error('sltoolbox:invaliddims', 'K0 should be a square matrix');
end

% for options
opts.preserve = {};
opts.weights = [];
opts.centralized = false;
opts = slparseprops(opts, varargin{:});

if isempty(opts.weights)
    isweighted = false;
else
    isweighted = true;
    if ~isequal(size(opts.weights), [1, n0])
        error('sltoolbox:sizmismatch', ...
            'The weights should be a 1 x n0 row vector');
    end
end


%% compute

%% centralize the gram matrix
if ~opts.centralized
    K0 = slcenkernel(K0, [], opts.weights);
end

%% solve the eigen-problem

if ~isweighted      % non weighted case    
    [evs, A] = slsymeig(K0);
    evs = evs(1:n0-1);
    
    k = sldim_by_eigval(evs, opts.preserve{:});
    A = A(:, 1:k);
    A = slmulvec(A, 1 ./ sqrt(evs(1:k))', 2); % normalize 
    
    evs = evs / n0;
    
else                % weighted case    
    
    w = max(opts.weights, 0);
    sw = sqrt(w)'; % sw is a column vector
    K0w = slmulrowcols(K0, sw', sw);
    [evs, A] = slsymeig(K0w);
    evs = evs(1:n0-1);
    
    k = sldim_by_eigval(evs, opts.preserve{:});
    A = A(:, 1:k);
    A = slmulrowcols(A, 1 ./ sqrt(evs(1:k))', sw); % normalize         
    
    evs = evs(1:n0-1) / sum(w);    
end
    
