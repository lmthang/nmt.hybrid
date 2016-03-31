function A = slaffinitymat(X, X2, nnparams, varargin)
%SLAFFINITYMAT Constructs an affinity matrix
%
% $ Syntax $
%   - A = slaffinitymat(X, [], nnparams, ...)
%   - A = slaffinitymat(X, X2, nnparams, ...)
%
% $ Arguments $
%   - X:        The sample matrix of the (source) nodes
%   - X2:       The sample matrix of the (target) nodes
%   - nnparams: The cell array of parameters for finding nearest neighbors
%               in the form of {method, ...}. Please refer to slfindnn
%               for details of specifying nnparams.
%   - A:        The constructed affinity matrix
%   
% $ Description $
%   - A = slaffinitymat(X, X2, nnparams, ...) constructs an affinity 
%     matrix to represent the pairwise affinity between neighboring 
%     samples. The affinity between non-neighboring samples is set to
%     zero. You can indicate a self-affinity matrix (affinity among the
%     the set of samples) by setting X2 to []. 
%     By default, the function uses heated kernel to compute the affinity
%     as explained below. In addition, you can use other formulas to 
%     translate the distances to affinity by supplying your tfunctor in 
%     the properties. 
%     The following is a table of properties that you can specify:
%       \*
%       \t   The Properties of Affinity Matrix construction         \\
%       \h     name         &      description                      \\
%             'sparse'      & Whether to construct sparse matrix 
%                             (default = true)                      \\
%             'kernel'      & The kernel to compute affinity         \\
%                             - 'heat': the heated kernel (default) 
%                               it uses the following formula to translate
%                               the Euclidean distances into affinities:
%                                 a = exp(- d^2 / (2 *sigma^2))
%                               Here sigma^2 is set to 
%                                 diffusion^2 * avg(d^2). 
%                               you can set the value of diffusion in the 
%                               properties.
%                             - 'simple':  the simple kernel
%                               just set the affinity of all neighboring
%                               samples to 1.
%             'diffusion'   & The diffusion distance relative to 
%                             the average distance. (default = 1)    \\
%             'tfunctor'    & The function to transform the distance
%                             values to edge values. (default = [])  \\
%             'sym'         & whether to symmetrizes the graph 
%                             (default = true)                       \\
%             'symmethod'   & The method used to symmetrization       
%                             (default = [])                          \\
%             'excludeself' & Whether to exclude the edges connecting
%                             the same node. (default = false).
%       \*
%
% $ Arguments $
%   - The properties sym, symmethod and excludeself only take effect
%     when input X2 = [].
%
% $ Remarks $
%   - It wrapps slnngraph to provide a convenient way to compute
%     typical affinity matrix.
%
% $ History $
%   - Created by Dahua Lin, on Sep 12nd, 2006
%

%% parse input and prepare parameters

if nargin < 3
    raise_lackinput('slaffinitymat', 3);
end

opts.sparse = true;
opts.kernel = 'heat';
opts.diffusion = 1;
opts.tfunctor = [];
opts.sym = true;
opts.symmethod = [];
opts.excludeself = false;
opts = slparseprops(opts, varargin{:});

if isempty(X2) 
    if opts.excludeself
        X2 = [];
    else
        X2 = X;
    end
else
    opts.sym = false;
end

if isempty(opts.tfunctor)
    switch opts.kernel
        case 'heat'
            tfunctor = {@internal_compaffinity, opts.diffusion};
        case 'simple'
            tfunctor = @(x) ones(size(x));
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid kernel name: %s', opts.kernel);
    end
else
    tfunctor = opts.tfunctor;
end

%% main wrapper

A = slnngraph(X, X2, nnparams, ...
    'valtype', 'numeric', ...
    'sparse', opts.sparse, ...
    'tfunctor', tfunctor, ...
    'sym', opts.sym, ...
    'symmethod', opts.symmethod);

%% The internal function to compute affinity

function affvals = internal_compaffinity(dists, diffusion)
    
sqs = dists .* dists;
sqs = sqs(:);

avgsq = sum(sqs) / length(sqs);
s = (diffusion^2) * avgsq * 2;

affvals = exp(-sqs / s);













