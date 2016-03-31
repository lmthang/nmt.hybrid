function GS = slgaussest(X, varargin)
%SLGAUSSEST Estimates the Gaussian models from samples
%
% $ Syntax $
%   - GS = slgaussest(X, ...)
%
% $ Arguments $
%   - X:        the sample matrix
%   - GS:       the gaussian model struct
%
% $ Description $
%   - GS = slgaussest(X, ...) estimates the parameters of Gaussian models
%     from the samples. There are three modes for estimation
%       - single mode:      only one model is estimated
%       - separated mode:   multiple models are trained on different
%                           sections of samples
%       - mixed mode:       multiple models are trained on all sections
%                           of samples, with different weights.
%
%     The following properties can be specified for 
%     the estimation:
%       - nums:         the numbers of samples in classes, default = [] 
%                       (for single mode or separated mode). It should be
%                       a 1 x k row vector.
%       - weights:      the weights of samples, default = []
%                       for single mode or separated mode: 1 x n row vector
%                       for mixed mode: k x n matrix
%       - compinv:      whether to compute the inverse of var/covar
%                       default = true.
%       - invparams:    the parameters for computation of var/covar
%                       it can be a cell of parameters shared by all models
%                       or a cell of cell arrays for different models 
%                       to have different parameters. default = {}
%       - varform:      the form of variance: 'univar'|'diagvar'|'covar'
%                       default = 'covar'.
%       - sharevar:     whether the variance/covariance is shared by 
%                       all models. default = false;
% 
% $ History $
%   - Created by Dahua Lin, on Aug 24, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - fix small redundant codes
%       - replace sladd by sladdvec and slmul by slmulvec to increase 
%         efficiency
%

%% parse and verify inputs

if ~isnumeric(X) || ndims(X) ~= 2
    error('sltoolbox:invalidarg', ...
        'X should be an numeric matrix');
end

[d, n] = size(X);
if n < 2
    error('sltoolbox:invalidarg', ...
        'More than one samples are needed for estimation');
end

opts.nums = [];
opts.weights = [];
opts.compinv = true;
opts.invparams = {};
opts.varform = 'covar';
opts.sharevar = false;
opts = slparseprops(opts, varargin{:});

weights = opts.weights;

% determine the mode
if isempty(opts.nums)
    if isempty(opts.weights)
        emode = 'single';
        k = 1;
    else
        if ndims(weights) ~= 2 || size(weights, 2) ~= n
            error('sltoolbox:sizmismatch', ...
                'The size of weights is inconsistent with that of samples');
        end
        k = size(weights, 1);
        if k == 1
            emode = 'single';
        else
            emode = 'mixed';
        end
    end
else
    nums = opts.nums;
    k = length(nums);
    if ~isequal(size(nums), [1, k])
        error('sltoolbox:invalidarg', ...
            'The nums should be a 1 x k row vector');
    end
    if sum(nums) ~= n
        error('sltoolbox:sizmismatch', ...
            'The nums is inconsistent with the number of samples');
    end    
    if k == 1
        emode = 'single';
    else
        emode = 'separated';
    end
    
    if ~isempty(weights)
        if ~isequal(size(weights), [1 n])
            error('sltoolbox:sizmismatch', ...
                'The size of weights is invalid for single/separated mode');
        end
    end
end

% check the var/covar

if ~ismember(opts.varform, {'univar', 'diagvar', 'covar'})
    error('sltoolbox:invalidarg', ...
        'The variance form %s is invalid', opts.varform);
end

invparams = opts.invparams;
if ~isempty(invparams)   
    if ~iscell(invparams)
        error('sltoolbox:invalidarg', ...
            'The invparams should be a cell array');
    end
    if iscell(invparams{1})        
        if ~isequal(size(invparams), [1 k])
            error('sltoolbox:sizmismatch', ...
                'The size of invparams is illegal');
        end
    end
end


%% Main skeleton for estimation
% for each varform, there are two corresponding functions
%   an estimation function for estimating single model
%   a merge function for merging new model to existing model struct
%

switch opts.varform
    case 'univar'        
        fh_est = @gaussest_univar;
        if ~opts.sharevar
            fh_merge = @gaussmerge_var;
        else
            fh_merge = @gaussmerge_combvar;
        end
    case 'diagvar'
        fh_est = @gaussest_diagvar;
        if ~opts.sharevar
            fh_merge = @gaussmerge_var;
        else
            fh_merge = @gaussmerge_combvar;
        end
    case 'covar'
        fh_est = @gaussest_covar;
        if ~opts.sharevar
            fh_merge = @gaussmerge_cov;
        else
            fh_merge = @gaussmerge_combcov;
        end            
end

switch emode
    case 'single'
        GS = fh_est(X, weights);
        
    case 'separated'
        pars = slpartition(n, 'blksizes', nums);
        GS = init_gaussmodels(d, k, opts.varform, opts.sharevar);     
        if ~isempty(weights)
            tw = sum(w);
        end
        for i = 1 : k
            si = pars.sinds(i); ei = pars.einds(i);
            curX = X(:, si:ei);
            if isempty(weights)
                curw = [];
                curportion = nums(i) / n;
            else
                curw = weights(si:ei);
                curportion = sum(curw) / tw;
            end            
            GSnew = fh_est(curX, curw);
            GS = fh_merge(GS, GSnew, i, curportion);
            clear GSnew;
        end
        
    case 'mixed'
        GS = init_gaussmodels(d, k, opts.varform, opts.sharevar); 
        tw = sum(sum(weights));
        for i = 1 : k
            curw = weights(i, :);
            curportion = sum(curw) / tw;
            GSnew = fh_est(X, curw);
            GS = fh_merge(GS, GSnew, i, curportion);
            clear GSnew;
        end
                
end

if opts.compinv
    switch opts.varform
        case {'univar', 'diagvar'}
            GS.invvars = slgaussinv(GS, 'vars', opts.invparams);
        case 'covar'
            GS.invcovs = slgaussinv(GS, 'covs', opts.invparams);
    end
end


%% Core routines for estimation

function GS = init_gaussmodels(d, k, varform, sharevar)

GS.dim = d;
GS.nmodels = k;
GS.means = zeros(d, k);

switch varform
    case 'univar'
        if ~sharevar            
            GS.vars = zeros(1, k);
        else
            GS.vars = 0;
        end
    case 'diagvar'
        if ~sharevar
            GS.vars = zeros(d, k);
        else
            GS.vars = zeros(d, 1);
        end
    case 'covar'
        if ~sharevar
            GS.covs = zeros(d, d, k);
        else
            GS.covs = zeros(d, d);
        end
end


function G = gaussest_univar(X, w)

vmean = slmean(X, w);
D = sladdvec(X, -vmean, 1);
D2 = D .* D;
clear D;
vars = sum(D2, 1);
[d, n] = size(X);
if isempty(w)
    vars = sum(vars) / (d * n);
else
    vars = sum(vars .* w) / (d * sum(w));
end

G.dim = d;
G.nmodels = 1;
G.means = vmean;
G.vars = vars;

function G = gaussest_diagvar(X, w)

vmean = slmean(X, w);
D = sladdvec(X, -vmean, 1);
D2 = D .* D;
clear D;
[d, n] = size(X);
if isempty(w)
    vars = sum(D2, 2) /  n;
else
    D2 = slmulvec(D2, w, 2);
    tw = sum(w);
    vars = sum(D2, 2) / tw;
end

G.dim = d;
G.nmodels = 1;
G.means = vmean;
G.vars = vars;

function G = gaussest_covar(X, w)

d = size(X, 1);
vmean = slmean(X, w);
C = slcov(X, w, vmean);

G.dim = d;
G.nmodels = 1;
G.means = vmean;
G.covs = C;


%% Core routines for model merging

function GS = gaussmerge_var(GS, g, i, p)

slignorevars(p);

GS.means(:,i) = g.means;
GS.vars(:,i) = g.vars;

function GS = gaussmerge_combvar(GS, g, i, p)

GS.means(:,i) = g.means;
GS.vars = GS.vars + p * g.vars;

function GS = gaussmerge_cov(GS, g, i, p)

slignorevars(p);

GS.means(:,i) = g.means;
GS.covs(:,:,i) = g.covs;

function GS = gaussmerge_combcov(GS, g, i, p)

GS.means(:,i) = g.means;
GS.covs = GS.covs + p * g.covs;


