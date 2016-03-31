function GS = slgausscomb(varargin)
%SLGAUSSCOMB Collects the means and variances/covariances to form GS
%
% $ Syntax $
%   - GS = slgausscomb('means', means, 'vars', vars, ...)
%   - GS = slgausscomb('means', means, 'covs', covs, ...)
%
% $ Arguments $
%   - means:        the mean vectors
%   - vars:         the variance values
%   - covs:         the covariance matrices
%   - GS:           the formed Gaussian model struct
%
% $ Description $
%   - GS = slgausscomb('means', means, 'vars', vars) forms the Gaussian 
%     model struct with varform being 'univar' or 'diagvar' using the mean
%     vectors and variance values.
%     The means can be given in either of the following forms:
%       - a d x k matrix
%       - a cell array with k cells, each cell being a d x 1 vector
%     The vars can be given in either of the following forms:
%       - a 1 x 1 scalar: for shared univar model
%       - a 1 x k vector: for non-shared univar model
%       - a d x 1 vector: for shared diagvar model
%       - a d x k matrix: for non-shared diagvar model
%       - a cell array of k scalars: for non-shared univar model
%       - a cell array of dx1 vectors: for non-shared diagvar model
%
%   - GS = slgausscomb('means', means, 'covs', covs) forms the Gaussian
%     model struct with varform being 'covar' using the mean vector and
%     covariance matrices.
%     The form of means is as mentioned above.
%     The covs can be given in either of the following forms:
%       - a d x d matrix: for shared covar model
%       - a d x d x k matrix: for non-shared covar model
%       - a cell array of k dxd matrices: for non-shared covar model
%
%     You can specify other properties to control the process
%       - 'invparams'  the cell array of parameters to compute inverse
%                      (default = {})
%                      For invvars, the computation is done by 
%                      slinvevals;
%                      For invcovs, the computation is done by
%                      slinvcovs;
%       - 'compinv'    whether to compute the inverse covariances
%                      (default = true)
%       - 'mixweights' the mixture weights (default = [])                        
%
% $ Remarks $
%   - You can specify either covs or vars, but you should not specify
%     both of them.
%
% $ History $
%   - Created by Dahua Lin, on Aug 24th, 2006
%

%% Take arguments

args.means = [];
args.vars = [];
args.covs = [];
args.compinv = true;
args.invparams = {};
args.mixweights = [];
args = slparseprops(args, varargin{:});


if isempty(args.means)
    error('sltoolbox:invalidarg', ...
        'The means should be specified');
end

if isempty(args.vars) && isempty(args.covs)
    error('sltoolbox:invalidarg', ...
        'You should specify either vars or covs');
end

if ~isempty(args.vars) && ~isempty(args.covs)
    error('sltoolbox:invalidarg', ...
        'You should specify both vars and covs');
end


%% Parse means

means = take_arrayform('means', args.means, 2);
[d, k] = size(means);

GS.dim = d;
GS.nmodels = k;
GS.means = means;


%% Parse variances / covariances

if ~isempty(args.vars)

    vars = take_arrayform('vars', args.vars, 2);
    [dv, kv] = size(vars);
    
    if dv ~= 1 && dv ~= d
        error('sltoolbox:sizmismatch', ...
            'The size of vars is illegal');
    end    
    if kv ~= 1 && kv ~= k
        error('sltoobox:sizmismatch', ...
            'The size of vars is illegal');
    end        
    
    GS.vars = vars;
    if args.compinv
        GS.invvars = slgaussinv(GS, 'vars', args.invparams);
    end
            
else
   
    covs = take_arrayform('covs', args.covs, 3);
    [dcv, dcv2, kcv] = size(covs);
    
    if dcv ~= d || dcv2 ~= d
       error('sltoolbox:sizmismatch', ...
           'The size of covs is illegal');
    end
    
    if kcv ~= 1 && kcv ~= k
        error('sltoolbox:sizmismatch', ...
            'The size of covs is illegal');
    end
    
    GS.covs = covs;    
    if args.compinv
        GS.invcovs = slgaussinv(GS, 'covs', args.invparams);
    end
                    
end

%% For mix weights

if ~isempty(args.mixweights)
    
    mixweights = args.mixweights(:);
    if length(mixweights) ~= k
        error('sltoolbox:sizmismatch', ...
            'The length of mix weights is illegal');
    end
    
    GS.mixweights = mixweights;
    
end

    


%% Auxiliary functions

function V = take_arrayform(name, v, dmax)

if isnumeric(v)
    V = v;
elseif iscell(v)
    V = v(:)';
    if dmax == 2
        V = horzcat(V{:});
    elseif dmax == 3
        V = cat(3, V{:});
    end
else
    error('sltoolbox:invalidarg', ...
        'The %s should be either an numeric array or a cell array', name);
end

if ndims(V) > dmax
    error('sltoolbox:invalidarg', ...
        'The dimension of means should not exceed %d', dmax);
end











