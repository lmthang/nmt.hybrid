function [models, info] = slproglearn(source, getter, learnfunctor, varargin)
%SLPROGLEARN Performs Progressive Learning from sample source
%
% $ Syntax $
%   - models = slproglearn(source, getter, learnfunctor, ...)
%   - [models, info] = slproglearn(source, getter, learnfunctor, ...)   
%
% $ Arguments $
%   - source:       the source from which the data are fetched
%   - getter:       the functor to fetch data from source, in the form:
%                   data= f(source, ...)
%   - learnfunctor: the functor to construct/update models from data,
%                   it is given in the following form:
%                   models = f(models, data, ...)
%                   If the learning is recorded, it is like:
%                   [models, record] = f(models, data, ...)
%                   On initial construction, the input models is [].
%   - models:       The constructed models
%   
% $ Description $
%   - models = slproglearn(source, getter, learnfunctor, ...) 
%     construct models based on source, from which the samples are
%     fetched. It is assumed that the source can continuously offer
%     infinite number of samples. The construction is controlled by 
%     the following properties.
%     \*
%     \t   Table. Properties of Progressive Learning
%     \h       name        &            description             \\
%          'isrecorded'    &  Whether each iteration is recorded 
%                             (default = false)                 \\
%          'gtuner'        &  The functor to tune the getter, in the form:
%                             getter = f(getter, models, ...)   
%                             (default = {})                      \\
%          'cmpfunctor'    &  The functor to compare two set of models
%                             and judge whether discrepancies meet the
%                             criteria of convergence.
%                             It is given in the following form:
%                             isconverged = f(models1, models2, ...)
%                             This properties must be specified.     \\
%          'iter'          &  The iteration control properties for
%                             sliterproc, default = {}.
%          'verbose'       &  whether to show progress information 
%                             (default = true)                       \\
%          'initmodels'    &  The initial models, (default = [])
%     \*
%
%   - [models, info] = slproglearn(source, getter, learnfunctor, ...) also
%     returns the information of iterative process.
%
% $ History $
%   - Created by Dahua Lin, on Aug 31, 2006
%

%% Parse and verify input arguments

if nargin < 3
    raise_lackinput('slproglearn', 3);
end

opts.isrecorded = false;
opts.gtuner = [];
opts.cmpfunctor = [];
opts.iter = {};
opts.verbose = true;
opts.initmodels = [];
opts = slparseprops(opts, varargin{:});

if isempty(opts.cmpfunctor)
    error('sltoolbox:invalidarg', ...
        'You should specify a models comparison functor');
end


%% Main Skeleton

slsharedisp_attach('slproglearn', 'show', opts.verbose);

slsharedisp(opts, 'Progressive Learning from source');

objects = {source, opts.initmodels, getter};
iterfunctor = {@proglearn_iter, learnfunctor, opts};
cmpfunctor = {@proglearn_cmp, opts};
if nargout < 2
    objects = ...
        sliterproc(objects, iterfunctor, cmpfunctor, opts.isrecorded, opts.iter{:});
else
    [objects, info] = ...
        sliterproc(objects, iterfunctor, cmpfunctor, opts.isrecorded, opts.iter{:});
end

models = objects{2};

slsharedisp_detach();



%% Core Iteration function

function varargout = proglearn_iter(objects, learnfunctor, opts)
% objects = {source, models, getter}

% take input
source = objects{1};
models = objects{2};
getter = objects{3};

% fetch data
data = slevalfunctor(getter, source);

% learn models
if ~opts.isrecorded
    models = slevalfunctor(learnfunctor, models, data);
else
    [models, rec] = slevalfunctor(learnfunctor, models, data);
end

% tune getter
if ~isempty(opts.gtuner)
    getter = slevalfunctor(opts.gtuner, getter, models);
end

% make output
objects = {source, models, getter};
if ~opts.isrecorded
    varargout = {objects};
else
    varargout = {objects, rec};
end


function isconverged = proglearn_cmp(objects_prev, objects, opts)

models_prev = objects_prev{2};
models = objects{2};

isconverged = slevalfunctor(opts.cmpfunctor, models_prev, models);






