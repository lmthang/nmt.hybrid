function arrMean = slarrmean(data, arrsiz, n, varargin)
%SLARRMEAN Computes the mean of a set of arrays
%
% $ Syntax $
%   - slarrmean(arrs, arrsiz, n, ...)
%   - slarrmean(fns, arrsiz, n, ...)
%
% $ Arguments $
%   - arrs:         the super-array consisting of all arrays
%   - fns:          the file paths of all array files
%   - arrsiz:       the size of each array unit. (for a column vector, it
%                   is the length of the vector)
%   - n:            the total number of array units
%
% $ Description $
%   - slarrmean(arrs, arrsiz, n, ...) computes the mean of all array units
%     with the size of each unit specified by arrsiz. If there are more 
%     than one array unit, the size of arrs should be [arrsiz, n].
%
%   - slarrmean(fns, arrsiz, n, ...) computes the mean of all array units
%     stored in the array files given in fns. Each array file stores
%     an super-array of a set of array units.
%
%   - You can specify additional properties.
%       \t      The properties of slarrmean
%       \h      name       &        description
%              'weights'   &  The weights of each array unit, default = []
%             
% $ History $
%   - Created by Dahua Lin on Jul 27th, 2006
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slarrmean', 3);
end
    
if isnumeric(data)
    isdirect = true;
    arrs = data;
    arrsiz = arrsiz(:)';
    if ~isequal(size(arrs), [arrsiz, n])
        error('sltoolbox:sizmismatch', ...
            'The size of arrs (data) is invalid');
    end
    
elseif iscell(data)
    isdirect = false;
    fns = data;
    arrsiz = arrsiz(:)';
    nfiles = numel(fns);
    
else
    error('sltoolbox:invalidarg', ...
        'The first argument for slarrmean should be an numeric array or a cell array of file names');
end

opts.weights = [];

opts = slparseprops(opts, varargin{:});
hasweights = ~isempty(opts.weights);

if (hasweights)
    opts.weights = opts.weights(:);
    if length(opts.weights) ~= n
        error('sltoolbox:invalidarg', ...
            'The length of weights is inconsistent with the number of units');
    end
end


%% Main skeleton

if isdirect
    arrMean = compute_array_sum(arrs, arrsiz, n, opts.weights);
else
    arrMean = zeros(arrsiz);
    c = 0;          
    for i = 1 : nfiles
        curarrs = slreadarray(fns{i});
        curn = size(curarrs, length(arrsiz) + 1);
        if ~hasweights
            arrMean = arrMean + compute_array_sum(curarrs, arrsiz, curn, []);
        else
            arrMean = arrMean + compute_array_sum(curarrs, arrsiz, curn, opts.weights(c+1:c+curn));
        end
        c = c + curn;
        
        if c > n
            error('sltoolbox:sizmismatch', ...
                'The total number of units in the set of array files is not n');
        end
    end
    
    if c ~= n
        error('sltoolbox:sizmismatch', ...
            'The total number of units in the set of array files is not n');
    end
    
end

if ~hasweights
    arrMean = arrMean / n;
else
    arrMean = arrMean / sum(opts.weights);
end



%% Compute function

function S = compute_array_sum(arrs, arrsiz, n, w)

if ~isequal(size(arrs), [arrsiz, n])
    error('sltoolbox:sizmismatch', ...
        'The size of array is not consistent as specified');
end

d = length(arrsiz);
if isempty(w)
    S = sum(arrs, d+1);
else
    S = reshape(arrs, [prod(arrsiz), n]) * w;
    if d == 1
        S = reshape(S, [arrsiz, 1]);
    else
        S = reshape(S, arrsiz);
    end
end










