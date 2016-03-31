function Y = slkernelfea(X, X0, kparams, varargin)
%SLKERNELFEA Extracts kernelized mapped features
%
% $ Syntax $
%   - Y = slkernelfea(X, X0, kparams)
%   - Y = slkernelfea(X, X0, kparams, ...)
%
% $ Arguments $
%   - X:            the target sample matrix.
%   - X0:           the referenced sample set.
%   - kparams:      the cell containing the parameters for kernel
%                   computation. 
%   - Y:            the feature matrix.
%
% $ Description $
%   - Y = slkernelfea(X, X0, kparams) computes the empirical kernel maps
%     for the samples X. kparams is a cell of parameters specifying
%     the computation of kernels, which is given in the form:
%     {kernel_type, ...} and input to slkernel function.
%
%   - Y = slkernelfea(X, X0, kparams, ...) computes kernel mapped features
%     according to the specified properties.
%     \*
%     \t   Table 1. Properties for Kernelized Feature Extraction \\
%     \h    name    &       description                          \\
%           'cen'   &  Whether to centralize the features. default = false.
%                      Note that, typically if centralization is applied
%                      in training stage, it should be also applied in
%                      testing stage for consistency.            \\
%           'proj'  &  The further projection coefficient matrix. 
%                      default = []. If an non-empty projection matrix
%                      is given, it will takes a further projection step.
%                      The projection matrix is of size n0 x d, where n0
%                      is the number of the referenced samples (i.e. the
%                      dimension of empirical kernel mapping), d is the 
%                      dimension of projected subspace.          \\
%           'gram'  &  The gram matrix of the referenced sample set. It
%                      will be used for centralization. When used, if it
%                      is not specified, the function will compute it from
%                      X0. However, it is more efficient to offer it
%                      when it is available and centralization is 
%                      required. default = []. \\
%         'weights' &  The weights of referenced samples. If specified, it
%                      will be used by centralization for mean feature
%                      computation. default = []. \\
%           'kfunc' &  The function for kernel computing. By default, it 
%                      is set to empty, which indicates to use slkernel
%                      for kernel computing. The user can supply its
%                      owned kernel computing function, which should
%                      follow the syntax as f(X0, X, ...). \\
%     \*
%
% $ History $
%   - Created by Dahua Lin on May 2nd, 2005
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slkernelfea', 3);
end

% for X
if ndims(X) ~= 2
    error('sltoolbox:invaliddims', ...
        'The sample matrix X should be a 2D matrix');
end
d = size(X, 1);

% for X0
if ndims(X0) ~= 2
    error('sltoolbox:invaliddims', ...
        'The sample matrix X0 should be a 2D matrix');
end
if size(X0, 1) ~= d
    error('sltoolbox:sizmismatch', ...
        'Size inconsistency between X and X0');
end
n0 = size(X0, 2);

% for kparams
if ~iscell(kparams)
    error('sltoolbox:invalidarg', ...
        'kernel parameters should be given by cell array');
end

% for options
opts.cen = false;
opts.proj = [];
opts.gram = [];
opts.kfunc = [];
opts.weights = [];
opts = slparseprops(opts, varargin{:});

% for projection matrix
if isempty(opts.proj)
    need_proj = false;
else
    if ndims(opts.proj) ~= 2
        error('sltoolbox:invaliddims', ...
            'The projection coefficient matrix should be 2D');
    end
    need_proj = true;
    A = opts.proj;
    if size(A, 1) ~= n0
        error('sltoolbox:sizmismatch', ...
            'Size inconsistency between X0 and A, the projection matrix A should be an n0 x d matrix');
    end
end

% for gram matrix K0
if isempty(opts.gram)
    has_gram = false;
else
    has_gram = true;
    K0 = opts.gram;
    if ~isequal(size(K0), [n0, n0])
        error('sltoolbox:sizmismatch', ...
            'Invalid size of gram matrix, which should be n0 x n0');
    end
end

% for kernel computing function
if ...
        isempty(opts.kfunc) || ...
        isequal(opts.kfunc, @slkernel) || ...
        strcmpi(opts.kfunc, 'slkernel')
    
    use_special_kfunc = false;
else
    use_special_kfunc = true;
end

% for sample weights
if ~isempty(opts.weights)
    if ~isequal(size(opts.weights), [1, n0])
        error('sltoolbox:sizmismatch', ...
            'The weights should be a 1 x n0 row vector');
    end
end


%% Compute

%% Empirical Kernel Mapping

if ~use_special_kfunc
    Kx = slkernel(X0, X, kparams{:});
else
    Kx = feval(opts.kfunc, X0, X, kparams{:});
end

%% Centralization

if opts.cen
    
    % compute gram matrix
    if ~has_gram
        if ~use_special_kfunc
            K0 = slkernel(X0, kparams{:});
        else
            K0 = feval(opts.kfunc, X0, X0, kparams{:});
        end                    
    end
    
    % centralize
    Kx = slcenkernel(K0, Kx, opts.weights);    
end

%% Projection and Output

if need_proj    
    Y = A' * Kx;     
else
    Y = Kx;
end


