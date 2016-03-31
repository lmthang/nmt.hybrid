function [GS, pp, info] = slgmm(X, varargin)
%SLGMM Learns Gaussian Mixture model from samples
%
% $ Syntax $
%   - GS = slgmm(X, ...)
%   - [GS, pp] = slgmm(X, ...)
%   - [GS, pp, info] = slgmm(X, ...)
%
% $ Arguments $
%   - GS:       The Gaussian model struct with mixture weights
%   - pp:       the posteriori of samples to component models
%   - info:     the information of learning process
%   - X:        the sample matrix 
%
% $ Description $
%   - GS = slgmm(X, ...) learns Gaussian mixture model from samples
%     in matrix X. The following properties can be specified:
%     \*  
%     \t    Table. Properties of Learning Gausssian Mixture Model
%     \h    name         &      description                      \\
%          'method'      & The method using for GMM learning. Currently,
%                          there are only one method available: 'EM',
%                          default = 'EM'.
%          'K'           & The number of initial components in the mixture 
%                          model (default = 3)                     \\
%          'update'      & The way of updating (default = 'pass'):
%                          1. 'pass'     Pass-wise update;
%                          2. 'comp'     Component-wise update     \\
%          'cyclecn'     & The ratio of components to be updated in each 
%                          cycle for 'comp' update scheme. default = 1.
%          'maxiter'     & The maximum number of iterations 
%                          (default = 100) \\
%          'tol'         & The maximum tolerance of posteriori error when 
%                          the iteration terminates, (default = 1e-6) \\
%          'verbose'     & whether to display information while iteration, 
%                          default = true                              \\
%          'annthres'    & The threshold of annealing in FJ algorithm, 
%                          default = 0 (unit = average mixture weight) \\
%          'weights'     & The weights of the samples, default = [], 
%                          indicating non-weighted. Weights should be given
%                          by a 1 x n row vector. \\
%          'varform'     & The form of variance: 'univar'|'diagvar'|'covar', 
%                          default = 'covar';          \\
%          'sharevar'    & Whether the variance is shared by all models,
%                          default = false;            \\
%          'invparams'   & The parameters for computing inverse of variance
%                          default = {}                \\
%           
% $ Remarks $
%   - In current version, the component-wise update scheme is only
%     supported when sharevar is false.
%
% $ History $
%   - Created by Dahua Lin, on Aug 28, 2006
%

%% parse and verify input arguments

if ~isnumeric(X) || ndims(X) ~= 2
    error('sltoolbox:invalidarg', ...
        'X should be a 2D numeric matrix');
end

opts.method = 'EM';
opts.K = 3;
opts.update = 'pass';
opts.cyclecn = 1;
opts.maxiter = 100;
opts.tol = 1e-6;
opts.verbose = true;
opts.annthres = 0;
opts.weights = [];
opts.varform = 'covar';
opts.sharevar = false;
opts.invparams = {};
opts = slparseprops(opts, varargin{:});

n = size(X, 2);


%% initialization by random clustering

if opts.K > n
    error('sltoolbox:sizoverflow', ...
        'The K is larger than the number of samples');
end

init_centerinds = randsample(n, opts.K);
init_centers = X(:, init_centerinds);
initc = slclassify_eucnn(init_centers, X);

%% perform learning based on FMM

estfunctor = {@gmm_estfunc, opts.varform, opts.sharevar, opts.invparams};
evalfunctor = {@gmm_evalfunc};

[GS, cw, pp, info] = slfmm(X, n, estfunctor, evalfunctor, ...
    'method', opts.method, ...
    'update', opts.update, ...
    'cyclecn', opts.cyclecn, ...
    'iter', {'maxiter', opts.maxiter}, ...
    'tol', opts.tol, ...
    'verbose', opts.verbose, ...
    'initc', initc, ...
    'annthres', opts.annthres, ...
    'weights', opts.weights, ...
    'estmode', 'innermul', ...
    'condpmode', 'log', ...
    'manifunc', @gmm_manifunc);

%% post actions

GS.mixweights = cw;


%% Core slot functions

function GS = gmm_estfunc(GS, X, n, W, selinds, varform, sharevar, invparams)

slignorevars(n);

if isempty(selinds)
    GS = slgaussest(X, 'weights', W, ...
        'varform', varform, 'sharevar', sharevar, ...
        'compinv', true, 'invparams', invparams);
else
    if sharevar
        error('sltoolbox:rterror', ...
            'The selective updating scheme is only supported when sharevar is false');
    end
    
    Wsel = W(selinds, :);
    GSu = slgaussest(X, 'weights', Wsel, ...
        'varform', varform, 'sharevar', sharevar, ...
        'compinv', true, 'invparams', invparams);
    
    nu = length(selinds);
    switch varform
        case 'univar'
            for idx = 1 : nu
                iu = selinds(idx);
                GS.means(:, iu) = GSu.means(:,idx);
                GS.vars(iu) = GSu.vars(idx);
                GS.invvars(iu) = GSu.invvars(idx);
            end
        case 'diagvar'
            for idx = 1 : nu
                iu = selinds(idx);
                GS.means(:, iu) = GSu.means(:, idx);
                GS.vars(:, iu) = GSu.vars(:, idx);
                GS.invvars(:, iu) = GSu.invvars(:, idx);
            end
        case 'covar'
            for idx = 1 : nu
                iu = selinds(idx);
                GS.means(:, iu) = GSu.means(:, idx);
                GS.covs(:,:, iu) = GSu.covs(:,:, idx);
                GS.invcovs(:,:, iu) = GSu.invcovs(:,:, idx);
            end
    end
    
end

function condp = gmm_evalfunc(GS, X, n)

slignorevars(n);
condp = slgausspdf(GS, X, 'output', 'log');


function GS = gmm_manifunc(GS0, op, varargin)

switch op
    case 'select'
        
        selinds = varargin{1};
        
        tyi = slgausstype(GS0);
        
        GS.dim = GS0.dim;
        GS.nmodels = length(selinds);
        GS.means = GS0.means(:, selinds);
        
        switch tyi.varform
            case {'univar', 'diagvar'}
                if tyi.sharevar
                    GS.vars = GS0.vars;
                    GS.invvars = GS0.invvars;
                else
                    GS.vars = GS0.vars(:, selinds);
                    GS.invvars = GS0.invvars(:, selinds);
                end
            case 'covar'
                if tyi.sharevar
                    GS.covs = GS0.covs;
                    GS.invcovs = GS0.invcovs;
                else
                    GS.covs = GS0.covs(:,:,selinds);
                    GS.invcovs = GS0.invcovs(:,:,selinds);
                end
        end
        
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Unsupported manipulation option for GMM: %s', op);
end







