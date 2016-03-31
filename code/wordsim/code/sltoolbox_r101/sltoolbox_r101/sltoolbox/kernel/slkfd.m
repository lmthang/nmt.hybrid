function A = slkfd(K, nums, varargin)
%SLKFD Perform Kernelized Fisher Discriminant Analysis
%
% $ Syntax $
%   - A = slkfd(K, nums, ...)
%
% $ Arguments $
%   - K:        the kernel gram matrix of the training samples
%   - nums:     the numbers of samples in all classes
%   - A:        the projection coefficient matrix 
% 
% $ Description $
%   - A = slkfd(K, nums, ...) performs Kernerlized Fisher discriminant 
%     analysis on the samples X according to the specified properties. 
%     \*
%     \t   Table 1.  The properties of Fisher Discriminant Analysis   \\
%     \h     name     &     description                                \\
%           'sol'     &  The cell containing the arguments for solving
%                        the generalized eigen-problem by slsymgeig.  
%                        default = {}.                                 \\
%           'dimset'  &  The cell containing the arguments for determining
%                        the output feature dimension. default = {}.
%                        (refer to sldim_by_eigval).                   \\
%           'Sb'      &  The pre-computed kernelized between-class 
%                        scattering matrix or the cell containing 
%                        the arguments for computing the kernelized 
%                        scatter matrix in the form {type, ...}, 
%                        which is input to slkernelscatter.     \\
%           'Sw'      &  The pre-computed kernelized within-class 
%                        scattering matrix or the cell containing 
%                        the arguments for computing the scatter 
%                        matrix in the form {type, ...}, which is 
%                        input to slkernelscatter.     \\
%         'weights'   &  The sample weights. default = [].   \\
%     \*  
%
% $ History $
%   - Created by Dahua Lin on May 03, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slkfd', 2);
end

% for K
n = size(K, 1);
if ~isequal(size(K), [n, n])
    error('sltoolbox:invaliddims', ...
        'K should be a square matrix');
end

% for nums
nc = length(nums);
if ~isequal(size(nums), [1 nc])
    error('sltoolbox:sizmismatch', ...
        'nums should be a 1 x nc row vector');
end

% for options

opts.sol = {};
opts.dimset = {};
opts.Sb = {'Sb'};
opts.Sw = {'Sw'};
opts.weights = [];
opts = slparseprops(opts, varargin{:});

has_Sb = ~isempty(opts.Sb) && isnumeric(opts.Sb);
has_Sw = ~isempty(opts.Sw) && isnumeric(opts.Sw);
if has_Sb
    Sb = opts.Sb;
    if ~isequal(size(Sb), [n n])
        error('sltoolbox:sizmismatch', ...
            'Sb should be a n x n matrix');
    end
end
if has_Sw
    Sw = opts.Sw;
    if ~isequal(size(Sw), [n n])
        error('sltoolbox:sizmismatch', ...
            'Sw should be a n x n matrix');
    end
end

if ~isempty(opts.weights)
    w = opts.weights;
    if ~isequal(size(w), [1 n])
        error('sltoolbox:sizmismatch', ...
            'The weights should be a 1 x n row vector');
    end
else
    w = [];
end


%% Compute 

%% Step 1: Construct the eigen-problem

if ~has_Sb
    Sb = slscatter(K, opts.Sb{:}, 'sweights', w, 'nums', nums);
end

if ~has_Sw
    Sw = slscatter(K, opts.Sw{:}, 'sweights', w, 'nums', nums);
end


%% Step 2: Resolve the eigen-problem

[evs, A] = slsymgeig(Sb, Sw, opts.sol{:});

rk = sldim_by_eigval(evs, opts.dimset{:});
A = A(:, 1:rk);


