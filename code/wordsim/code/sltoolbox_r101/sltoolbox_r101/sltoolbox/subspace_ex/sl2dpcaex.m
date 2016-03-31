function [Mm, PL, PR, info] = sl2dpcaex(data, matsiz, n, method, varargin)
%SL2DPCAEX Learns Extended 2D PCA on a set of matrix samples
%
% $ Syntax $
%   - [Mm, PL, PR] = sl2dpcaex(data, matsiz, n, method, ...)
%   - [Mm, PL, PR, info] = sl2dpcaex(data, matsiz, n, method, ...)
%
% $ Arguments $
%   - data:         the matrix samples (or the array files storing them)
%   - matsiz:       the size of the matrices in terms of [nrows, ncols]
%   - n:            the number of samples
%   - method:       the method used for pursue the optimal projections
%   - Mm:           the mean image
%   - PL:           the left projection matrix
%   - PR:           the right projection matrix
%   - info:         the learning process information
%
% $ Description $
%   - [PL, PR] = sl2dpcaex(data, matsiz, n, method, ...) performs extended 
%     2D PCA on a set of matrix samples given in data. The data can be an
%     nrows x ncols x n array or a cell array of array filenames.
%     matsiz and n respectively specify the sample size and sample number.
%     The learning process will finally output the left and right
%     projection matrices for reducing the column and row space.
%     \*
%     \t    Table 1. The methods for optimal projection pursuit       \\
%     \h     name       &    description                              \\
%           'dimfix'    & The dimension is fixed during optimization. \\
%           'grareduce' & Gradually reduce the dimension in a dynamic 
%                         process.                                    \\
%     \*
%     For each method, you can specify additional properties to control
%     the learning process.
%     \*
%     \t   Table 2. The properties for dimfix learning        \\
%     \h    name      &       description                     \\
%          'tarsiz'   &  The fixed target feature size (required)
%          'maxiter'  &  The maximum number of iterations
%                        (default = 50)                       \\
%          'tol'      &  The maximum allowance in the change of objective
%                        function in terms of the ratio of the initial 
%                        objective value (default = 1e-6)     \\
%          'weights'  &  The weights of samples (default = []) \\ 
%          'verbose'  &  Whether to show iteration information  
%                        (default = true)                      \\
%     \*
%     \*
%     \t   Table 3. The properties for grareduce learning      \\
%     \h    name      &       description                      \\
%          'eprate'   &  The energy preservation rate in each iteration  
%                         (default = 0.998)                     \\
%          'er'       &  The ratio of energy to be preserved finally 
%                         (default = 0.98)                      \\
%          'weights'  &  The weights of samples                 \\
%          'verbose'  &  Whether to show iteration information  \\
%                        (default = true)
%     \*
%
% $ History $
%   - Created by Dahua Lin, on Jul 30th, 2006
%

%% Parse and Verify Input Arguments
if nargin < 4
    raise_lackinput('sl2dpcaex', 4);
end
matsiz = matsiz(:)';
if length(matsiz) ~= 2
    error('sltoolbox:invalidarg', ...
        'The processing matrices should be 2D, thus the matsiz should be a two-element vector');
end
if ~isnumeric(data) && ~iscell(data)
    error('sltoolbox:invalidarg', ...
        'The data should be an numeric array or a cell array of filenames');
end
check_datasize(data, matsiz, n);


%% Main Sketelon

switch method
    case 'dimfix'
        [Mm, PL, PR, info] = do2dpca_dimfix(data, matsiz, n, varargin{:});
        
    case 'grareduce'
        [Mm, PL, PR, info] = do2dpca_grareduce(data, matsiz, n, varargin{:});
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid method for 2D PCA: %s', method);
end


%% Strategy functions

%% dimfix Strategy

function [Mm, PL, PR, info] = do2dpca_dimfix(data, matsiz, n, varargin)

% parse and verify options

opts.tarsiz = [];
opts.maxiter = 100;
opts.tol = 1e-6;
opts.weights = [];
opts.verbose = true;
opts = slparseprops(opts, varargin{:});

if ~isequal(size(opts.tarsiz), [1, 2])
    error('sltoolbox:invalidarg', ...
        'The target size should be a 1 x 2 row vector and it should be specified');
end
if any(opts.tarsiz > matsiz)
    error('sltoolbox:sizmismatch', ...
        'Some dimensions of target size is larger than matsize');
end

k1 = opts.tarsiz(1);
k2 = opts.tarsiz(2);
w = opts.weights;

% main body

showinfo('Start 2D PCA using Strategy: dimfix.', opts);

% initialization
showinfo('Initialization ...', opts);
PL = eye(matsiz(1));
PR = eye(matsiz(2));
Mm = slarrmean(data, matsiz, n, 'weights', opts.weights);

S0 = sl2dmatcov('CL', data, matsiz, n, Mm, [], [], w);
totalenergy = trace(S0);
energycurve = [];
clear S0;

% alternate update
showinfo('Alternate Updating ...', opts);
isconverged = false;
it = 0;
while(~isconverged && it < opts.maxiter)
    
    % start an iteration
    it = it + 1;
    
    % update
    SL = sl2dmatcov('CL', data, matsiz, n, Mm, [], PR, w);
    PL = solve_proj_dimfix(SL, k1);
    SR = sl2dmatcov('CR', data, matsiz, n, Mm, PL, [], w);
    PR = solve_proj_dimfix(SR, k2);
    
    % monitor objective
    curenergy = trace(PR' * SR * PR);    
    if it > 1
        isconverged = (abs(curenergy - energycurve(end)) < totalenergy * opts.tol);
    end
    energycurve = [energycurve; curenergy]; 
    
    % summary iteration
    itermsg = sprintf('Iter %4d: energy preservation = %f', it, curenergy / totalenergy);
    showinfo(itermsg, opts);       
    
end

showinfo('2D PCA completed.', opts);

% output info

info.total_energy = totalenergy;
info.energy_curve = energycurve;
info.numiters = it;
info.energy_ratio = curenergy / totalenergy;
info.is_converged = isconverged;

    
%% grareduce Strategy

function [Mm, PL, PR, info] = do2dpca_grareduce(data, matsiz, n, varargin)

% parse and verify options

opts.eprate = 0.998;
opts.er = 0.98;
opts.weights = [];
opts.verbose = true;
opts = slparseprops(opts, varargin{:});

w = opts.weights;

% main body

showinfo('Start 2D PCA using Strategy: grareduce.', opts);

% initialization
showinfo('Initialization ...', opts);
PL = eye(matsiz(1));
PR = eye(matsiz(2));
Mm = slarrmean(data, matsiz, n, 'weights', opts.weights);

S0 = sl2dmatcov('CL', data, matsiz, n, Mm, [], [], w);
totalenergy = trace(S0);
energycurve = [];
dimcurve = [];
clear S0;

% dynamic evolve
it = 0;
curenergy = totalenergy;

while curenergy > totalenergy * opts.er
    
    % start an iteration
    it = it + 1;
    
    % update
    SL = sl2dmatcov('CL', data, matsiz, n, Mm, PL, PR, w);
    PLm = solve_proj_energypreserve(SL, size(PL, 2), opts.eprate);
    PL = PL * PLm;
    SR = sl2dmatcov('CR', data, matsiz, n, Mm, PL, PR, w);
    PRm = solve_proj_energypreserve(SR, size(PR, 2), opts.eprate);
    PR = PR * PRm;
        
    % monitor objective
    curenergy = trace(PRm' * SR * PRm);
    clear PLm PRm;
    energycurve = [energycurve, curenergy]; 
    dimcurve = [dimcurve; [size(PL, 2), size(PR, 2)]];
    
    % summary iteration
    itermsg = sprintf('Iter %4d: siz = [%d, %d], energy preservation = %f', ...
        it, size(PL, 2), size(PR, 2), curenergy / totalenergy);
    showinfo(itermsg, opts);     
    
end

showinfo('2D PCA completed.', opts);

% output info

info.total_energy = totalenergy;
info.dim_curve = dimcurve;
info.energy_curve = energycurve;
info.numiters = it;
info.energy_ratio = curenergy / totalenergy;




%% Core computing function


function P = solve_proj_dimfix(S, k)

[evals, evecs] = slsymeig(S);

if k > length(evals)
    error('k is larger than the number of eigenvalues');
end

P = evecs(:, 1:k);


function P = solve_proj_energypreserve(S, kmax, r)

[evals, evecs] =  slsymeig(S);

k = min(sum(cumsum(evals) < sum(evals) * r), kmax);
P = evecs(:, 1:k);




%% Auxiliary functions


function check_datasize(data, matsiz, n)

if isnumeric(data)
    if (n == 1 && ~isequal(size(data), matsiz)) || ...
            (n > 1 && ~isequal(size(data), [matsiz, n]))
        error('sltoolbox:sizmismatch', ...
            'The size of data does not match the specified matsize and sample number');
    end
else
    nfiles = length(data);
    cf = 0;
    for i = 1 : nfiles
        curarr = slreadarray(data{i});
        curn = size(curarr, 3);
        check_datasize(curarr, matsiz, curn);
        cf = cf + curn;
        clear curarr;
    end
    if cf ~= n
        error('sltoolbox:sizmismatch', ...
            'The size of data does not match the specified matsize and sample number');
    end
end


function showinfo(message, opts)

if opts.verbose
    disp(message);
end


