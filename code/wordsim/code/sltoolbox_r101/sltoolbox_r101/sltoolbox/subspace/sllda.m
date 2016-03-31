function T = sllda(X, nums, method, varargin)
%SLLDA Trains a Linear Discriminant Model using specified method
%
% $ Syntax $
%   - T = sllda(X, nums, method, ...)
%
% $ Arguments $
%   - X:        the sample matrix, with each column representing a sample
%   - nums:     the numbers of samples in all classes
%   - method:   the selected method for training
%   - T:        the trained LDA model
%
% $ Description $
%   - T = sllda(X, nums, method, ...) trains a LDA transform using
%     specified method. It is actually a wrapper of some underlying
%     LDA training functions such as slfld, sldlda, and slnlds etc,
%     and provides a more friendly interface for users.
%       
%     \*
%     \t    Table 1. The methods for LDA Training
%     \h    name      &       description 
%          'pinv'     & using pseudo inverse to invert to Sw
%          'efm'      & using enhanced fisher model, with following params 
%          'boundev'  & bounding the eigenvalues
%          'regdual'  & the dual-space LDA based on simple regularization
%          'pvldual'  & the dual-space LDA based on PVL
%          'nlda'     & the null-space LDA
%          'dlda'     & the direct LDA
%     \*          
%
%     You can specify addtional parameters to control the training
%     process as follows:
%
%     \*
%     \t    Table 2. The LDA Training properties
%     \h    name      &         description
%           'prepca'  &  whether to perform a preceding PCA step
%                        default = false, (for all methods except for dlda)
%           'rvalue'  &  The r-value for whitening, typically it is the
%                        minimum ratio of effective eigenvalue to the
%                        largest eigenvalue. 
%                        default = [], that is to leave the corresponding
%                        method to decide the default value.
%           'dimset'  &  the params to determine the number of output 
%                        features, default = {'rank'}. 
%                        You should use a cell to encompass the parameters
%                        fed to sldim_by_eigval.
%           'Sb'      &  The pre-computed between-class scatter matrix
%                        or the cell array of params to compute Sb by
%                        slscatter. default = {'Sb'}
%           'Sw'      &  The pre-computed between-class scatter matrix
%                        or the cell array of params to compute Sb by
%                        slscatter. default = {'Sw'}
%           'pdimset' &  The params to determine the range space.
%                        (only for nlda and dlda, default = {}).
%           'weights' &  The sample weights, default = [];
%     \*
%
% $ History $
%   - Created by Dahua Lin, on Aug 16th, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('sllda', 3);
end

opts.prepca = false;
opts.rvalue = [];
opts.dimset = {'rank'};
opts.Sb = {'Sb'};
opts.Sw = {'Sw'};
opts.weights = [];
opts.pdimset = {};

opts = slparseprops(opts, varargin{:});

%% delegate to concrete functions

switch method
    case 'pinv'
        whitenopt = {'std', eps};
        T = delegate_fld(X, nums, opts, whitenopt);
    case 'efm'
        r = take_value(opts.rvalue, 1e-5);
        whitenopt = {'std', r};
        T = delegate_fld(X, nums, opts, whitenopt);
    case 'boundev'
        r = take_value(opts.rvalue, 1e-3);
        whitenopt = {'bound', r};
        T = delegate_fld(X, nums, opts, whitenopt);
    case 'regdual'
        r = take_value(opts.rvalue, 1e-3);
        whitenopt = {'reg', r};
        T = delegate_fld(X, nums, opts, whitenopt);
    case 'pvldual'
        r = take_value(opts.rvalue, 2e-3);
        whitenopt = {'gapprox', r};
        T = delegate_fld(X, nums, opts, whitenopt);
    case 'nlda'
        T = delegate_nlda(X, nums, opts);
    case 'dlda'
        T = delegate_dlda(X, nums, opts);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid LDA method %s', method);
end


%% Delegation wrapper

function T = delegate_fld(X, nums, opts, whitenopt)

ropts.prepca = opts.prepca;
ropts.whiten = {'scheme', 'std', 'evproc', whitenopt};
ropts.dimset = opts.dimset;
ropts.Sb = opts.Sb;
ropts.Sw = opts.Sw;
ropts.weights = opts.weights;

T = slfld(X, nums, ropts);


function T = delegate_nlda(X, nums, opts)

ropts.prepca = opts.prepca;
ropts.pdimset = opts.pdimset;
ropts.dimset = opts.dimset;
ropts.Sb = opts.Sb;
ropts.Sw = opts.Sw;
ropts.weights = opts.weights;

T = slnlda(X, nums, ropts);

function T = delegate_dlda(X, nums, opts)

ropts.pdimset = opts.pdimset;
ropts.Sb = opts.Sb;
ropts.Sw = opts.Sw;
ropts.weights = opts.weights;

T = sldlda(X, nums, ropts);


%% Auxiliary function

function v = take_value(v0, defaultv)

if isempty(v0)
    v = defaultv;
else
    v = v0;
end

