function [models, Q, info] = slreevallearn(models, Q, data, estfunctor, evalfunctor, cmpfunctor, varargin)
%SLREEVALLEARN Performs an iterative learning based on re-evaluation
%
% $ Syntax $
%   - [models, Q] = slreevallearn(models, data, estfunctor, evalfunctor, cmpfunctor, ...)
%   - [models, Q, info] = slreevallearn(models, data, estfunctor, evalfunctor, cmpfunctor, ...)
%
% $ Arguments $
%   - models:       the models to learn
%                   input one: the initial model
%                   output one: the learned model
%   - Q:            the evaluated quantities on the data
%   - data:         the data (samples) to learn from
%   - estfunctor:   the functor to estimate/update model, in the form:
%                   models = f(models, data, Q, ...)
%                   If the process is recorded, it is like:
%                   [models, rec] = f(models, data, Q, ...)
%   - evalfunctor:  the functor to evaluate/update the quantities, like:
%                   Q = f(models, data, Q, ...)
%   - cmpfunctor:   the functor to judge convergence, like:
%                   isconverged = f(models_prev, models, Q_prev, Q, ...)
%   - info:         The information of the iteration process
%
% $ Description $
%   - [models, Q] = slreevallearn(models, data, estfunctor, evalfunctor, cmpfunctor, ...) 
%     implements a skeleton of learning procedure based on re-evaluation.
%     The procedure alternates the model estimation based on data and 
%     corresponding evaluated quantities, and the evaluation by applying
%     models to data to get quantities.
%     \*
%     \t   Table. The Properties of Re-evaluating Learning
%     \h    name        &   description
%          'iter'       & The iteration control properties. default = {}
%          'isrecorded' & whether the model estimation is recorded
%                         default = false;
%          'verbose'    & whether to show progress, default = true
%     \*
%   
% $ Remarks $
%   - It is implemented based on sliterproc.
%
%   - It essentially serves as the core of many learning paradigms, 
%     such as E-M method of finite mixture model learning, AdaBoost,
%     K-means, robust subspace learning, etc.
%
% $ History $
%   - Created by Dahua Lin, on Aug 31, 2006
%

%% parse and verify input

if nargin < 6
    raise_lackinput('slreevallearn', 6);
end

opts.iter = {};
opts.isrecorded = false;
opts.verbose = true;
opts = slparseprops(opts, varargin{:});

%% Main 

slsharedisp_attach('slreevallearn', 'show', opts.verbose);

slsharedisp('Learning by re-evaluation');

objects = {models, data, Q};
iterfunctor = {@reevallearn_iter, estfunctor, evalfunctor, opts};
cmpfunctor = {@reevallearn_cmp, cmpfunctor};
if nargout < 2
    objects = ...
        sliterproc(objects, iterfunctor, cmpfunctor, opts.isrecorded, opts.iter{:});
else
    [objects, info] = ...
        sliterproc(objects, iterfunctor, cmpfunctor, opts.isrecorded, opts.iter{:});
end

models = objects{1};
Q = objects{3};

slsharedisp_detach();


%% Iteration functors

function varargout = reevallearn_iter(objects, estfunctor, evalfunctor, opts)
% objects = {models, data, Q}

[models, data, Q] = deal(objects{:});

if ~opts.isrecorded
    models = slevalfunctor(estfunctor, models, data, Q);
else
    [models, info] = slevalfunctor(estfunctor, models, data, Q);
end
Q = slevalfunctor(evalfunctor, models, data, Q);

objects = {models, data, Q};

if ~opts.isrecorded
    varargout = {objects};
else
    varargout = {objects, info};
end


function isconverged = reevallearn_cmp(objects_prev, objects, cmpfunctor)

models_prev = objects_prev{1};
Q_prev = objects_prev{3};
models = objects{1};
Q = objects{3};

isconverged = slevalfunctor(cmpfunctor, models_prev, models, Q_prev, Q);











