function T = slnlda(X, nums, varargin)
%SLNLDA Performs Nullspace-based Linear Discriminant Analysis
%
% $ Syntax $
%   - T = slnlda(X, nums)
%   - T = slnlda(X, nums, ...)
%
% $ Arguments $
%   - X:        the training sample matrix
%   - nums:     the numbers of samples in all classes
%   - T:        the solved transform matrix
%
% $ Description $
%   - T = slnlda(X, nums) performs nullspace LDA on the samples X using 
%     default settings.
%
%   - T = slnlda(X, nums, ...) performs nullspace LDA on the samples X
%     with the specified properties.
%     \*
%     \t   Table 1.  The properties of Fisher Discriminant Analysis   \\
%     \h     name    &     description                                \\
%           'prepca' &  Whether to perform a preamble PCA to first 
%                       reduce the dimensions to the samples' rank.
%                       default = false.                              \\
%           'pdimset' &  The cell containing the arguments for determining
%                        the dimension of the principal subspace (that is
%                        the orthogonal complement of the nullspace)   \\
%           'dimset'  &  The cell containing the arguments for determining
%                        the output feature dimension. default = {}.
%                        (refer to sldim_by_eigval).                   \\
%           'Sb'      &  The pre-computed between-class scattering matrix
%                        or the cell containing the arguments for 
%                        computing the scatter matrix in the form
%                        {type, ...}, which is input to slscatter.     \\
%           'Sw'      &  The pre-computed within-class scattering matrix
%                        or the cell containing the arguments for 
%                        computing the scatter matrix in the form
%                        {type, ...}, which is input to slscatter.     \\
%         'weights'   &  The sample weights. default = [].             \\
%     \*  
%
% $ Remarks $
%   -# The function solves the transform in mainly following stages: 
%      First, solve the null space of the between-class scattering, then
%      project all samples onto the null space. Finally, a PCA-step is 
%      conducted to maximize the between-class scattering on nullspace.
%      
%   -# If Sw or its computing rule is given, the null space is  directly 
%      solved from Sw, otherwise the null space is solved from within class 
%      differences. If Sb is given, the between-class scattering on null 
%      space is computed by directly applying the null space projection
%      to Sb, otherwise, Sb is computed from components on nullspace. If 
%      both Sb and Sw are given, then the samples are not used in the 
%      function. In this cases, you can simply input an empty X. 
%
%   -# If both Sb and Sw are given, the pre-pca step will not be conducted.
%      no matter whether prepca is true or false.
%
% $ History $
%   - Created by Dahua Lin on May 1st, 2006
%   - Modified by Dahua Lin on Sep 10th, 2006
%       - replace sladd by sladdvec and slmul by slmulvec to increase 
%         efficiency
%


%% parse and verify input arguments 

if nargin < 2
    raise_lackinput('slfld', 2);
end

% check size

if ~isempty(X)    
    if ndims(X) ~= 2
        error('sltoolbox:invaliddims', ...
            'The sample matrix X should be a 2D matrix');
    end
    [d, n] = size(X);
    
    k = length(nums);
    if ~isequal(size(nums), [1, k]);
        error('sltoolbox:invaliddims', ...
            'The nums vector should be a row vector');
    end
    if sum(nums) ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number in nums is not consistent with that in X');
    end
end

% check options

opts.prepca = false;
opts.pdimset = {};
opts.dimset = {};
opts.Sb = {'Sb'};
opts.Sw = {'Sw'};
opts.weights = [];
opts = slparseprops(opts, varargin{:});

has_Sb = ~isempty(opts.Sb) && isnumeric(opts.Sb);
has_Sw = ~isempty(opts.Sw) && isnumeric(opts.Sw);
if has_Sb && has_Sw
    use_samples = false;
    d = size(opts.Sw, 1);
    
    if ~isequal(size(opts.Sb), [d, d]) || ~isequal(size(opts.Sw), [d, d])
        error('sltoolbox:sizmismatch', ...
            'Size consistency in Sb and Sw');
    end
        
else
    if isempty(X)
        error('sltoolbox:invalidargs', ...
            'The samples cannot be empty when Sb or Sw is not pre-computed');
    end
    use_samples = true;
    if (has_Sb && ~isequal(size(opts.Sb), [d, d])) || (has_Sw && ~isequal(size(opts.Sw), [d, d]))
        error('sltoolbox:sizmismatch', ...
            'Size consistency in Sb and Sw');
    end
    
end
w = opts.weights;


%% Compute

%% Step 0: Pre-PCA
pca_computed = false;
if use_samples && opts.prepca
    SPCA = slpca(X, 'weights', w);
    X = SPCA.P' * sladdvec(X, -SPCA.vmean, 1);
    pca_computed = true;
end

%% Step 1: Solve Null Space

if has_Sw
    PN = slnullspace({'cov', opts.Sw}, opts.pdimset{:});
elseif ~isempty(opts.Sw) && ~isequal(opts.Sw, {'Sw'})
    Sw = slscatter(X, opts.Sw{:}, 'sweights', w, 'nums', nums);
    PN = slnullspace({'cov', Sw}, opts.pdimset{:});
    clear Sw;
else 
    PN = slnullspace(make_weighted_withinclass_diffvecs(X, w, nums), ...
        opts.pdimset{:});
end

if pca_computed
    T1 = SPCA.P * PN;
    clear SPCA PN;
else
    T1 = PN;
    clear PN;
end


%% Step 2: Compute the second-stage transform

if has_Sb
    WSb = T1' * opts.Sb * T1;
else
    X = T1' * X;
    WSb = slscatter(X, opts.Sb{:}, 'sweights', w, 'nums', nums);
end
[evs, T2] = slsymeig(WSb);
rk2 = sldim_by_eigval(evs, opts.dimset{:});
T2 = T2(:, 1:rk2);

%% Integrate the transforms

T = T1 * T2;


%% The function for making the weighted difference vectors 
function Y = make_weighted_withinclass_diffvecs(X, w, nums)

mvs = slmeans(X, w, nums);
Y = X;
[sp, ep] = slnums2bounds(nums);
k = length(nums);
for i = 1 : k
    Y(:, sp(i):ep(i)) = sladdvec(X(:, sp(i):ep(i)), -mvs(:,i), 1);
end

if ~isempty(w)
    Y = slmulvec(Y, sqrt(max(w, 0)), 2);
end

