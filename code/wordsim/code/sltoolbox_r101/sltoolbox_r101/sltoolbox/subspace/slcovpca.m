function S = slcovpca(vmean, C, preserve)
%SLCOVPCA Trains a PCA with the covariance matrix given
%
% $ Syntax $
%   - S = slcovpca(vmean, C)
%   - S = slcovpca(vmean, C, preserve)
%
% $ Arguments $
%   - vmean:    the mean vector
%               (set vmean to zero indicates a zero mean vector)
%   - C:        the covariance matrix
%   - preserve: the scheme of determinaton of the subspace dimension
%               default = {'rank'}
%   - S:        the struct of PCA model
%
% $ History $
%   - Created by Dahua Lin, on Aug 17, 2006
%


[evals, evecs] = slsymeig(C);

evals = max(evals, 0);
if nargin < 3 || isempty(preserve)
    k = sldim_by_eigval(evals);
else
    k = sldim_by_eigval(evals, preserve{:});
end

d = size(C, 1);
total_energy = sum(evals);
if k < d
    evals = evals(1:k);
    evecs = evecs(:, 1:k);
    prin_energy = sum(evals);
else
    prin_energy = total_energy;
end


S.sampledim = d;
S.feadim = k;
S.support = [];
if isequal(vmean, 0)
    S.vmean = zeros(d, 1);
else
    S.vmean = vmean;
end
S.P = evecs;
S.eigvals = evals;
S.residue = total_energy - prin_energy;
S.energyratio = prin_energy / total_energy;






