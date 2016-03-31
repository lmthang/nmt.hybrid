function X = slgaussrnd(GS, nums)
%SLGAUSSRND Generates random samples from Gaussian models
%
% $ Syntax $
%   - X = slgaussrnd(GS, nums)
%
% $ Arguments $
%   - GS:       The Gaussian model struct
%   - nums:     the number of samples from the models
%
% $ Description $
%   - X = slgaussrnd(GS, nums) randomly draws samples from Gaussian models.
%     Suppose there are k constituent models. Then nums can be of
%     the following forms:
%       - length-k vector, representing the number of samples of the
%         models respectively.
%
% $ History $
%   - Created by Dahua Lin, on Aug 24, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - Replace sladd by sladdvec to increase efficiency
%

%% parse and verify input arguments

tyinfo = slgausstype(GS);
d = GS.dim;
k = GS.nmodels;

if length(nums) ~= k
    error('sltoolbox:invalidarg', ...
        'The nums should be a length-k vector');
end

n = sum(nums);
ps = slpartition(n, 'blksizes', nums);

%% generate samples

X = zeros(d, n);

switch tyinfo.varform
    case 'univar'
        if tyinfo.sharevar
            for i = 1 : k
                si = ps.sinds(i); ei = ps.einds(i);
                X(:, si:ei) = gensamples_univar(GS.means(:,i), GS.vars, d, nums(i));
            end
        else
            for i = 1 : k
                si = ps.sinds(i); ei = ps.einds(i);
                X(:, si:ei) = gensamples_univar(GS.means(:,i), GS.vars(i), d, nums(i));
            end                
        end
        
    case 'diagvar'
        if tyinfo.sharevar
            for i = 1 : k
                si = ps.sinds(i); ei = ps.einds(i);
                X(:, si:ei) = gensamples_diagvar(GS.means(:,i), GS.vars, d, nums(i));
            end
        else
            for i = 1 : k
                si = ps.sinds(i); ei = ps.einds(i);
                X(:, si:ei) = gensamples_diagvar(GS.means(:,i), GS.vars(:,i), d, nums(i));
            end                
        end
        
    case 'covar'
        if tyinfo.sharevar
            for i = 1 : k
                si = ps.sinds(i); ei = ps.einds(i);
                X(:, si:ei) = gensamples_covar(GS.means(:,i), GS.covs, d, nums(i));
            end
        else
            for i = 1 : k
                si = ps.sinds(i); ei = ps.einds(i);
                X(:, si:ei) = gensamples_covar(GS.means(:,i), GS.covs(:,:,i), d, nums(i));
            end                
        end
        
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid variance form in GS: %s', tyinfo.varform);
        
end


%% Core generation routines

function X = gensamples_univar(vmean, var, d, n)

X = randn(d, n) * sqrt(max(var, 0));
X = sladdvec(X, vmean, 1);

function X = gensamples_diagvar(vmean, vars, d, n)

X = randn(d, n);
X = slmulvec(X, sqrt(max(vars, 0)), 1);
X = sladdvec(X, vmean, 1);

function X = gensamples_covar(vmean, C, d, n)

X = randn(d, n);
[D, V] = slsymeig(C);
T = V * diag(sqrt(max(D, 0)));
clear D V;
X = T * X;
clear T;
X = sladdvec(X, vmean, 1);











