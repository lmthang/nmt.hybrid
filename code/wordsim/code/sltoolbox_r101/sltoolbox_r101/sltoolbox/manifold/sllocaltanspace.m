function [LM, LP, LS] = sllocaltanspace(X0, G, dl) 
%SLLOCALTANSPACE Solves the local tangent spaces
%
% $ Syntax $
%   - [LM, LP] = sllocaltanspace(X0, G, dl)
%   - [LM, LP, LS] = sllocaltanspace(...)
%
% $ Arguments $
%   - X0:       The referenced sample matrix (d0 x n0)
%   - G:        The neighborhood graph (n0 x n)
%   - dl:       The dimension of local tangent spaces
%               (should be strictly less than d0 and n0)
%   - LM:       The local means (d0 x n)
%   - LP:       The local tangent space basis (d0 x dl x n)
%   - LS:       The local spectrum (dl x n)
%
% $ Description $
%   - [LM, LP] = sllocaltanspace(X0, G, dl) solves the local tangent
%     spaces based on the neighborhood graph G. Suppose G is n0 x n,
%     (n0 source points and n target points), then it solves the local
%     tangent spaces at n target points with the space constructed with
%     their neighbors in X0. If G is valued, the values in G are the 
%     weights of the samples in constructing local tangent space.
%
%   - [LM, LP, LS] = sllocaltanspace(...) additionally outputs the
%     eigen-spectrum of the local spaces. 
%
% $ Remarks $
%   - The local dimensions are sorted in descending order of the 
%     corresponding eigenvalues. In the case of local rank < dl,
%     for the last dl - rank dimensions, the eigenvalues are set to
%     zeros, and the eigenvectors are set to zero vectors.
%
% $ History $
%   - Created by Dahua Lin, on Sep 13rd, 2006
%

%% parse and verify input arguments

if ~isnumeric(X0) || ndims(X0) ~= 2
    error('sltoolbox:invalidarg', ...
        'The sample matrix X0 should be a 2D numeric matrix');
end
[d0, n0] = size(X0);

gi = slgraphinfo(G);
if gi.n ~= n0
    error('sltoolbox:invalidarg', ...
        'The graph is not consistent with the sample number');
end
if ~strcmp(gi.form, 'adjmat')
    G = sladjmat(G, 'sparse', true);
end
n = gi.nt;

if dl >= d0 || dl >= n0
    error('sltoolbox:invalidarg', ...
        'The local dimension dl should be strictly less than d0 and n0');
end

if nargout >= 3
    want_sp = true;
else
    want_sp = false;
end

if isnumeric(G)
    use_weights = true;
else
    use_weights = false;
end

%% main skeleton

% prepare storage
LM = zeros(d0, n);
LP = zeros(d0, dl, n);
if want_sp
    LS = zeros(dl, n);
end

% do computation
for i = 1 : n    
    localinds = find(G(:,i));
    if use_weights
        localw = G(localinds, i)';
    else
        localw = [];
    end
    localX = X0(:, localinds);
    
    [cm, cp, csp] = solvelocalspace(localX, localw, d0, dl);
    
    LM(:,i) = cm;
    LP(:,:,i) = cp;
    if want_sp
        LS(:,i) = csp;
    end
end


%% core routine to compute local tangent spaces

function [vmean, P, spectrum] = solvelocalspace(X, w, d0, dl)

n = size(X, 2);
dm = min([dl, d0, n]);

% preprocess samples: centralize and weight
vmean = slmean(X, w, true);
X = sladdvec(X, -vmean, 1);
if ~isempty(w)
    X = slmulvec(X, w, 2);
end

% solve eigen-problem
if d0 <= n / 2
    C = X * X';
    [spectrum, P] = slsymeig(C, dm); 
else
    C = X' * X;
    [spectrum, P] = slsymeig(C, dm);
    P = slnormalize(X * P);
end

% truncate to rank
rk = sum(spectrum > n * eps(spectrum(1)));
if rk < dm
    spectrum = spectrum(1:rk);
    P = P(:, 1:rk);    
end
spectrum = spectrum / n;

% complement to dl
if rk < dl
    spectrum = [spectrum; zeros(dl-rk, 1)];
    P = [P, zeros(d0, dl-rk)];
end

