function S = slscatter(X, type, varargin)
%SLSCATTER Compute the scatter matrix 
%
% $ Syntax $
%   - S = slscatter(X, type, ...)
%
% $ Arguments $
%   - X:        the sample matrix with each column representing a sample
%   - type:     the type of scatter matrix to compute
%   - S:        the resulting scatter matrix 
%
% $ Description $
%   - S = slscatter(X, type, ...) computes the scatter matrix for the 
%     samples in X of specific type according to the properties specified.
%     \*
%     \t    Table 1. The types of scatter matrix          \\
%     \h    name   &     description                      \\
%           'Sw'   &  Within class scatter matrix. That is to compute
%                     scatter matrix within the samples in each class, and
%                     sum up the class-specific scatters. \\
%           'Sb'   &  Between class scatter matrix. That is to compute
%                     scatter matrix on the means of the classes. And uses
%                     the number of samples in each class or the total
%                     sample weight for each class as the weight for
%                     the mean vectors.                   \\
%           'St'   &  Total scatter matrix. That is to compute the scatter
%                     matrix on all the samples. The class-specific info.
%                     is ignored for St.                  \\
%     \*
%     
%     You can specify following properties to customize the computation
%     of the scatter matrix.
%     \*
%     \t    Table 2. The properties of scatter computation       \\
%     \h    name      &      description                         \\
%          'method'   &  The method of computation, can be 'std' or 'pw'.
%                        'std' computes the scatter in standard manner
%                        S = sum_i w_i (x_i - mv)(x_i - m_v)';
%                        'pw' computes the scatter in pairwise manner
%                        S = sum_{ij} w_{ij} (x_i - x_j) (x_i - x_j)'; 
%                        default = [].                           \\
%          'nums'     &  The numbers of samples in all classes. These 
%                        numbers are used for groupping the samples for
%                        'Sw' and 'Sb'. It is ignored for 'St'. If 
%                        it is set to empty, all samples are considered
%                        from the same class. default = [].      \\
%          'sweights' &  The weights of samples. Suppose there are n
%                        samples. Then sweights should be an 1 x n row
%                        vector. default = [].                   \\
%          'dwrule'   &  The rule for determining the weights from 
%                        distances. It should be an invokable object 
%                        determining the weights from distances either
%                        from samples to corresponding class centers
%                        for 'std' method or between samples of pairs 
%                        for 'pw' method. default = {}.           \\
%     \*
%
% $ Remarks $
%   -# The invokable object for determining weights from distances has
%      two forms. In the non-parametric form, it can be a function name,
%      function handle, or an inline object, the array of distances would 
%      be the unique argument input. In the parametric form, it
%      is given in a cell {f, ...}, f can be a function name, function
%      handle, or an inline object. The array of distances followed by
%      the additonal arguments in the cell will be input to f. 
%           
%   -# For weighting, if not specified all weights are considered to be 1.
%      In 'std' method, the weights w_i for each term is given by 
%      the multiplication of sweights(i) and the weights determined by
%      dwrule. In 'pw' method, the weights w_{ij} for each term is 
%      given by sweights(i) * sweights(j) * the value determined by dwrule.
%
%   -# The computation in 'std' method is based on slcov and its result
%      will be scaled by n or total weight. The computation in 'pw' method
%      is based on slpwscatter.
%
% $ History $
%   - Created by Dahua Lin on Apr 27, 2005
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slscatter', 2);
end

% check sample matrix X

if ndims(X) ~= 2
    error('sltoolbox:invaliddims', ...
        'The sample matrix X should be a 2D matrix');
end
[d, n] = size(X);

% check type

if ~ismember(type, ...
        {'Sw', 'Sb', 'St'})
    error('sltoolbox:invalidarg', ...
        'Invalid type of scatter matrix: %s', type);
end

% check options

opts.method = 'std';
opts.nums = [];
opts.sweights = [];
opts.dwrule = {};

opts = slparseprops(opts, varargin{:});

switch opts.method
    case 'std'
        fh_scatter_core = @scatter_core_std;
        fh_get_weights = @get_weights_std;
    case 'pw'
        fh_scatter_core = @scatter_core_pw;
        fh_get_weights = @get_weights_pw;
    otherwise
        error('sltoolbox:invalidarg', ...
        'Invalid method for scatter computing: %s', opts.method);
end
        
    
if isempty(opts.nums)
    k = 1;
    opts.nums = n;
else
    k = length(opts.nums);
    if ~isequal(size(opts.nums), [1, k])
        error('sltoolbox:invalidarg', ...
            'nums should be an 1 x k row vector');
    end
    if sum(opts.nums) ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number is consistent with that in X');
    end
end

if ~isempty(opts.sweights) 
    if ~isequal(size(opts.sweights), [1, n])
        error('sltoolbox:invalidarg', ...
            'sample weights should be an 1 x n row vector.');
    end
end
    

%% Compute

switch type
    
%% Compute Sw
    case 'Sw'        
        if k == 1   % single class
            w = fh_get_weights(X, opts.sweights, opts.dwrule);
            S = fh_scatter_core(X, w);
        else        % multiple classes
            S = zeros(d, d);
            [sp, ep] = slnums2bounds(opts.nums);
            for i = 1 : k
                [curX, curw] = get_cur_class(X, sp(i), ep(i), opts, fh_get_weights);
                S = S + fh_scatter_core(curX, curw);
            end
        end                
        
%% Compute Sb        
    case 'Sb'
        if k == 1   % single class (no between class scattering)
            S = zeros(d, d);
        else        % multiple classes
            centers = slmeans(X, opts.sweights, opts.nums);
            [sp, ep] = slnums2bounds(opts.nums);
            sw = get_class_weights(opts.sweights, opts.nums, sp, ep);
            w = fh_get_weights(centers, sw, opts.dwrule);
            S = fh_scatter_core(centers, w);
        end    
                        
%% Compute St
    case 'St'        
        w = fh_get_weights(X, opts.sweights, opts.dwrule);
        S = fh_scatter_core(X, w);                
        
end
        
        


%% The function for computing weights
function w = get_weights_std(X, sweights, dwrule)

if isempty(sweights)    
    if isempty(dwrule)      % default weights       
        w = [];        
    else                    % weights from dwrule
        mv = slmean(X, [], true);
        dists = slmetric_pw(mv, X, 'eucdist');
        w = invoke_dwrule(dwrule, dists);
    end        
else
    if isempty(dwrule)      % weights from sweights
        w = sweights;
    else                    % both sweights and dwrule
        mv = slmean(X, sweights, true);
        dists = slmetric_pw(mv, X, 'eucdist');
        w = invoke_dwrule(dwrule, dists) .* sweights;
    end
end

       
function w = get_weights_pw(X, sweights, dwrule)

if isempty(sweights)
    if isempty(dwrule)      % default weights
        w = [];
    else                    % weights from dwrule
        dists = slmetric_pw(X, X, 'eucdist');
        w = invoke_dwrule(dwrule, dists);
    end
else
    if isempty(dwrule)      % weights from sweights
        w = sweights' * sweights;   
    else                    % both sweights and dwrule
        w = sweights' * sweights;
        dists = slmetric_pw(X, X, 'eucdist');
        w = invoke_dwrule(dwrule, dists) .* w;
    end
end
        
        
%% The auxiliary function to extracting a class of samples and weights
function [curX, curw] = get_cur_class(X, sp, ep, opts, fhgetw)

curX = X(:, sp:ep);
if isempty(opts.sweights)
    sw = [];
else
    sw = opts.sweights(sp:ep);
end
curw = fhgetw(curX, sw, opts.dwrule);


%% The auxiliary function to getting the class weights
function w = get_class_weights(sweights, nums, sp, ep)

if isempty(sweights)
    w = nums;
else
    k = length(nums);    
    w = zeros(1, k);
    for i = 1 : k
        w(i) = sum(sweights(sp(i):ep(i)));
    end
end
    


%% The core computing function for compute a single scattering

function S = scatter_core_std(X, w)

if isempty(w)
    n = size(X, 2);
    S = slcov(X) * n;
else
    tw = sum(w);
    S = slcov(X, w) * tw;
end


function S = scatter_core_pw(X, w)

S = slpwscatter(X, w);


        
%% The auxiliary function for computing weights from dwrule
function w = invoke_dwrule(dwrule, dists)

if ~iscell(dwrule)
    w = feval(dwrule, dists);
else
    if length(dwrule) == 1
        w = feval(dwrule{1}, dists);
    else
        w = feval(dwrule{1}, dists, dwrule{2:end});
    end
end


