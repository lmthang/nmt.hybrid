function T = sldlda(X, nums, varargin)
%SLDLDA Performs Direct Linear Discriminant Analysis
%
% $ Syntax $
%   - T = sldlda(X, nums)
%   - T = sldlda(X, nums, ...)
%
% $ Arguments $
%   - X:        the training sample matrix
%   - nums:     the numbers of samples in all classes
%   - T:        the solved transform matrix
%
% $ Description $
%   - T = sldlda(X, nums) performs direct LDA on the samples X using 
%     default settings.
%
%   - T = sldlda(X, nums, ...) performs direct LDA on the samples X
%     with the specified properties.
%     \*
%     \t   Table 1.  The properties of Fisher Discriminant Analysis   \\
%     \h     name     &     description                                \\
%           'pdimset' &  The cell containing the arguments for determining
%                        the range space of Sb. They will be input to
%                        slrangespace for dimension determination.
%           'whiten'  &  The cell containing the arguments for computing 
%                        the whitening transform in 2nd stage. They will
%                        input to slwhiten_from_cov.
%                        default = {}.       \\
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
%      First solves the range space of between-class scattering, 
%      projecting all samples onto it. Then, solve the whitening transform
%      of the projected within-class scattering.
%      
%   -# If Sb or its computing rule is given, the range space is  directly 
%      solved from Sb, otherwise the null space is solved from class 
%      centers. If both Sb and Sw are given, then the samples are not 
%      used in the function. In this cases, you can simply input an empty X. 
%
%   -# If both Sb and Sw are given, the pre-pca step will not be conducted.
%      no matter whether prepca is true or false.
%
% $ History $
%   - Created by Dahua Lin on May 1st, 2006
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
opts.pdimset = {};
opts.whiten = {};
opts.Sb = {'Sb'};
opts.Sw = {'Sw'};
opts.weights = [];
opts = slparseprops(opts, varargin{:});

has_Sb = ~isempty(opts.Sb) && isnumeric(opts.Sb);
has_Sw = ~isempty(opts.Sw) && isnumeric(opts.Sw);
if has_Sb && has_Sw
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
    if (has_Sb && ~isequal(size(opts.Sb), [d, d])) || (has_Sw && ~isequal(size(opts.Sw), [d, d]))
        error('sltoolbox:sizmismatch', ...
            'Size consistency in Sb and Sw');
    end
    
end
w = opts.weights;

%% Step 1: Compute range space of Sb

if has_Sb
    T1 = slrangespace({'cov', opts.Sb}, opts.pdimset{:});
elseif ~isempty(opts.Sb) && ~isequal(opts.Sb, {'Sb'})
    Sb = slscatter({'cov', X}, opts.Sb{:}, 'sweights', w, 'nums', nums);
    T1 = slrangespace(Sb, opts.pdimset{:});
    clear Sb;
else
    Xc = get_weighted_centers(X, w, nums);
    T1 = slrangespace(Xc, opts.pdimset{:});
    clear Xc wc;
end


%% Step 2: Compute the whiten transform for Sw on range space

if has_Sw
    PSw = T1' * opts.Sw * T1;
else
    X = T1' * X;
    PSw = slscatter(X, opts.Sw{:}, 'sweights', w, 'nums', nums);
end
T2 = slwhiten_from_cov(PSw, opts.whiten{:});
T2 = flipdim(T2, 2);

%% Integrate the transforms

T = T1 * T2;


%% The function for computing weighted centers
function [Xc, wc] = get_weighted_centers(X, w, nums)

Xc = slmeans(X, w, nums);
if isempty(w)
    wc = nums;
else
    k = length(nums);
    [sp, ep] = slnums2bounds(nums);
    wc = zeros(1, k);
    for i = 1 : k
        wc(i) = sum(w(sp(i):ep(i)));
    end
end
wc = sqrt(max(wc, 0));
Xc = slmulvec(Xc, wc, 2);


