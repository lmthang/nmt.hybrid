function [S, cw, pp, info] = slfmm(X, n, estfunctor, evalfunctor, varargin)
%SLFMM Learns a Finite Mixture Model (FMM)
%
% $ Syntax $
%   - [S, cw] = slfmm(X, n, estfunctor, evalfunctor, ...)
%   - [S, cw, pp] = slfmm(X, n, estfunctor, evalfunctor, ...)
%   - [S, cw, pp, info] = slfmm(X, n, estfunctor, evalfunctor, ...)
% 
% $ Arguments $
%   - X:            the samples
%   - n:            the number of samples in X
%   - estfunctor:   the functor to estimate the model from weighted 
%                   samples, it should take the following form.
%                   The functor is a cell array, with the first element
%                   being a function, while the others are the extra
%                   parameters: {estfunc, ...}
%                   For different modes, the function should take different
%                   forms:
%                   - 'simple' mode:
%                       s = estfunc(s0, X, n, w, ...)
%                   - 'innermul' mode:
%                       S = estfunc(S0, X, n, W, selinds, ...)
%                   Here
%                   - s:    a single model
%                   - s0:   the previous single model (initially, it is [])
%                   - S:    a multi-model
%                   - S0:   the previous multi model (initially, it is [])
%                   - X:    the sample set
%                   - n:    the number of samples
%                   - w:    the 1 x n weight vector for a specific
%                           component
%                   - W:    the k x n weight matrix for all components
%                   - selinds:  the selected indices of models to be
%                               updated. If it is empty, all models need
%                               to be updated.
%   - evalfunctor:  the functor to evaluate component-conditional 
%                   likelihood
%                   condp = evalfunc(S, X, n, ...)
%                   for 'simple' mode, condp is a length-n vector
%                   for 'innermul', condp is k x n matrix.
%                   If condpmode is 'log', then the condp is logarithm of
%                   the probabilities.  
%   - S:        the struct of the learned finite mixtured model
%   - cw:       the component mixture weights
%   - pp:       the posteriori of each sample w.r.t mixture components
%
% $ Description $
%   - [S, cw, pp] = slfmm(X, n, estfunc, evalfunc, ...) learns the finite 
%     mixture model from the samples in X, according to the properties.
%     \*
%     \t  Table 1. Properties of GMM Learning                 \\
%     \h    name      &      description                      \\
%          'method'   & The method using for GMM learning. Currently,
%                       there are only one method available: 'EM',
%                       default = 'EM'.
%          'update'   & The way of updating (default = 'pass'):
%                        1. 'pass'     Pass-wise update;
%                        2. 'comp'     Component-wise update     \\
%          'cyclecn'  & The ratio of components to be updated in each 
%                       cycle for 'comp' update scheme. default = 1.
%          'iter'     & The iteration control properties for sliterproc
%                       default = {'maxiter', 100}
%          'tol'      & The maximum tolerance of posteriori error when 
%                       the iteration terminates, (default = 1e-6) \\
%          'verbose'  & whether to display information while iteration, 
%                       default = true                              \\
%          'initc'    & The labels of initial clustering, if not specified, 
%                       we will use random clustering. If initc is specified,
%                       K will be forced to the the number of unique labels.
%                       default = []. \\
%          'initpp'   & The initial posteriori of samples. (k0 x n)  
%                       default = []. \\
%          'annthres' & The threshold of annealing in FJ algorithm, 
%                       default = 0 (unit = average mixture weight) \\
%          'weights'  & The weights of the samples, default = [], 
%                       indicating non-weighted. Weights should be given
%                       by a 1 x n row vector. \\
%          'estmode'  & The mode of estimation:
%                       'simple':    S is a struct array of models, each
%                                    element represents a model, there is
%                                    no internal representation of 
%                                    component weights. (default)
%                       'innermul':  S is a struct or an object maintaining
%                                    a set of models
%          'condpmode' & The mode of evaluated likelihood
%                        'normal':  they are the likelihood values
%                        'log':     they are the logarithm of the
%                                   likelihood
%          'manifunc'  & The function to manipulate the models
%                        (only for innermul mode)
%                        S = manifunc(S0, 'select', [k1, k2, ...])
%                           to retain the k1-th, k2-th, ... models
%     \*
%
%   - [S, cw, pp, info] = slfmm(X, n, estfunc, evalfunc, ...)
%     also outputs the information of learning process. The info is a 
%     struct with following fields
%       - numiters:     the number of iterations
%       - stopdiscrep:  the discrepance value on stop
%       - isconverged:  whether the process has been converged
%
% $ Remarks $
%   - Note that there is no any restriction on the form of X. The
%     only condition is that the estfunc and evalfunc accept it.
%
%   - When annthres is larger than zero, the manifunc supporing
%     the operation 'discard' is required.
%
%   - For initialization, you should specify either of the initc or 
%     initpp. If both initc or initpp are given, only the initpp
%     will take effect.
%   
% $ History $
%   - Created by Dahua Lin on Aug 17, 2006
%       - as a foundation of all finite mixture models, such as GMM
%       - it is based on the original slgmm function
%   - Modified by Dahua Lin on Aug 29, 2006
%       - based on the sliterproc to control iteration process.
%   - Modified by Dahua Lin on Sep 10, 2006
%       - replace slmul by slmulvec to increase efficiency
%

%% parse and verify input arguments

props.method = 'EM';
props.update = 'pass';
props.cyclecn = 1;
props.iter = {'maxiter', 100};
props.tol = 1e-6;
props.verbose = true;
props.initc = [];
props.initpp = [];
props.annthres = 0;
props.weights = [];
props.estmode = 'simple';
props.condpmode = 'normal';
props.manifunc = [];

props = slparseprops(props, varargin{:});

% add two properties
props.estfunctor = estfunctor;
props.evalfunctor = evalfunctor;


checkvalid('learning method', props.method, {'EM'});
checkvalid('updating scheme', props.update, {'pass', 'comp'});
checkvalid('estimation mode', props.estmode, {'simple', 'innermul'});
checkvalid('conditional probability form', props.condpmode, {'normal', 'log'});

if isempty(props.initc) && isempty(props.initpp)
    error('sltoolbox:invalidarg', ...
        'Please specify either initc or initpp');
end

if ~isempty(props.weights)
    if ~isequal(size(props.weights), [1, n])
        error('sltoolbox:sizmismatch', ...
            'The weights should be a 1 x n row vector');
    end
end

if props.annthres > 0 
    if isempty(props.manifunc)
        error('sltoolbox:rterror', ...
            'The manipulation function is required when component annealing is on');
    end
end


%% initialization

slsharedisp_attach('slfmm', 'show', props.verbose);

% determine initial K value and initial clustering
if ~isempty(props.initpp)
    
    pp = props.initpp;
    W = weightmap(pp, props.weights);
    
elseif ~isempty(props.initc)
    
    [pp, W] = labels2weights(props.initc, props.weights);
        
end

initK = size(W, 1);

slsharedisp('FMM Learning Parameters:');
slsharedisp_incindent;
slsharedisp('method  = %s', props.method);
slsharedisp('update  = %s', props.update);
slsharedisp('estmode = %s', props.estmode);
slsharedisp('init K  = %d', initK);
slsharedisp('anneal thres = %g', props.annthres);
slsharedisp_decindent;

%% updating

switch props.method       
    case 'EM'
        slsharedisp('Run Finite-Mixture Model Learning by EM');
        
        em_estfunctor = {@fmm_em_est, props};
        em_evalfunctor = {@fmm_em_eval, props};
        em_cmpfunctor = {@fmm_em_cmp, props};
        
        models = {[], []};
        data = {X, n};
        Q = {pp, W};
        
        props.iter = {props.iter{:}, 'titlebreak', false};
        [models, Q, info] = slreevallearn(models, Q, data, ...
            em_estfunctor, em_evalfunctor, em_cmpfunctor, ...
            'iter', props.iter, 'isrecorded', false);
        
        S = models{1};
        cw = models{2};
        pp = Q{1};
    
end

slsharedisp_detach;


%% The Iteration functions for EM

% data = {X, n}
% Q = {pp, W}
% models = {S, cw}

function models = fmm_em_est(models, data, Q, props)

S = models{1};
X = data{1};  
% n = data{2};
W = Q{2};

if isempty(S)  % initialize model
    [S, cw] = init_models(X, W, props);
    
else           % update model  
    switch props.update
        case 'pass'
            [S, W, cw] = update_models(S, X, W, [], props);
            slignorevars(W);

        case 'comp'
            cn = max(ceil(props.cyclecn * size(W, 1)), 1);
            for t = 1 : cn
                si = ceil(rand * size(W,1));
                si = min(max(si, 1), size(W, 1));
                [S, W, cw] = update_models(S, X, W, si, props);
            end
    end
end

models = {S, cw};


function Q = fmm_em_eval(models, data, Q, props)

S = models{1}; cw = models{2};
X = data{1}; n = data{2};
slignorevars(Q);

[pp, W] = update_weightmap(S, cw, X, n, props);

Q = {pp, W};


function isconverged = fmm_em_cmp(models_prev, models, Q_prev, Q, props)

slignorevars(models_prev, models);

W_prev = Q_prev{2};
W = Q{2};
isconverged = false;

slsharedisp_attach('fmm_em_cmp');

if isequal(size(W), size(W_prev))
    wdiff = sldiscrep(W_prev, W, 'maxdiffnrm', true);
    slsharedisp('K = %d: wdiff = %g', size(W, 1), wdiff);
    if wdiff < props.tol
        isconverged = true;
    end
else
    slsharedisp('K = %d -> %d', size(W_prev, 1), size(W, 1));
end

slsharedisp_detach;




%% Core routines

% The function to initialize models
function [S, cw] = init_models(X, W, props)

[k, n] = size(W);
switch props.estmode
    case 'simple'
        for i = 1 : k
            S(i,1) = slevalfunctor(props.estfunctor, [], X, n, W(i, :)); 
        end        
    case 'innermul'
        S =  slevalfunctor(props.estfunctor, [], X, n, W, []);                      
end

cw = sum(W, 2);
cw = cw / sum(cw);


% The function to update posteriori (E-step)
function [pp, W] = update_weightmap(S, cw, X, n, props)

% compute conditional pdf
k = length(cw);
switch props.estmode
    case 'simple'
        condp = zeros(k, n);
        for i = 1 : k
            condp(i, :) = slevalfunctor(props.evalfunctor, S(i), X, n);
        end
    case 'innermul'
        condp = slevalfunctor(props.evalfunctor, S, X, n);
end

% compute posteriori
switch props.condpmode
    case 'normal'
        pp = slposteriori(condp, cw);
    case 'log'
        pp = slposteriori(condp, cw, 'log');
end

% compute weight map
W = weightmap(pp, props.weights);


% The function to update models (M-step)
function [S, W, cw] = update_models(S, X, W, selinds, props) 

% update weights
k0 = size(W, 1);
[S, W, cw] = update_compweights(S, W, props);
k1 = size(W, 1);
if k0 ~= k1
    return;
end

% update model parameters
[k, n] = size(W);
switch props.estmode
    case 'simple'
        if isempty(selinds)
            for i = 1 : k
                S(i,1) = slevalfunctor(props.estfunctor, S(i), X, n, W(i,:));
            end
        else
            ns = length(selinds);
            for ii = 1 : ns
                i = selinds(ii);
                S(i,1) = slevalfunctor(props.estfunctor, S(i), X, n, W(i,:));
            end
        end
    case 'innermul'        
        S = slevalfunctor(props.estfunctor, S, X, n, W, selinds);
end




%% Auxiliary functions

% The function to select weak components

function is_weak = select_weak_components(cw, thres)

wthres = thres / length(cw);
is_weak = (cw < wthres);

% The function anneal components

function [S, W, cw] = anneal_components(S, W, cw, props)

is_weak = select_weak_components(cw, props.annthres);
if all(is_weak)
    [maxw, si] = max(cw);
    slignorevars(maxw);
elseif any(is_weak)
    si = find(~is_weak);
else
    return;
end
       
switch props.estmode
    case 'simple'
        S = S(si);
    case 'innermul'
        S = feval(props.manifunc, S, 'select', si);
end

W = W(si, :);

cw = cw(si);
cw = cw / sum(cw);
    
% The function to update weights (possibly anneal components)

function [S, W, cw] = update_compweights(S, W, props)

cw = sum(W, 2);
cw = cw / sum(cw);

if props.annthres > 0
    [S, W, cw] = anneal_components(S, W, cw, props);
end

% The function to compute weights from labels

function [pp, W] = labels2weights(labels, sweights)

labelset = unique(labels);
[tf, inds] = ismember(labels, labelset);
slignorevars(tf);
clear tf;
k = length(labelset);
n = length(labels);

pp = zeros(k, n);
pp(((1:n) - 1) * k + inds) = 1;
W = weightmap(pp, sweights);


% The function to update posteriori to sample-component weights

function W = weightmap(pp, weights)

if isempty(weights)
    W = pp;
else
    W = slmulvec(pp, weights, 2);
end


%% Simple Auxiliary Functions

function checkvalid(name, val, range)

if ~ismember(val, range)
    error('sltoolbox:invalidarg', ...
        'Invalid %s: %s', name, val);
end



