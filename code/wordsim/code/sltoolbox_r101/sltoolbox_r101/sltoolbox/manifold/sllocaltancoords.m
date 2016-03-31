function [Y, GM] = sllocaltancoords(LM, LP, X, G, ga)
%SLLOCALTANCOORDS Computes the local tangent coordinates
%
% $ Syntax $
%   - [Y, GM] = sllocaltancoords(LM, LP, X, G)
%   - [Y, GM] = sllocaltancoords(LM, LP, X, G, ga);
%
% $ Arguments $
%   - LM:       The local means (d0 x m)
%   - LP:       The local space basis (d0 x dl x m)
%   - X:        The input samples (d0 x n)
%   - G:        The graph indicating which sample is managed by which model
%               (1 x n)
%   - ga:       The graph arrangement (default = 'N')
%                   - 'N': m x n, models as sources, samples as targets
%                   - 'T': n x m, samples as sources, models as targets
%   - Y:        The computed local coordinates (dl x nnz)
%
% $ Description $
%   - [Y, GM] = sllocaltancoords(LM, LP, X, G) computes the local tangent
%     coordinates using the default graph arrangement. For each non-zero
%     entry in G, there is a pairs of a space model characterized by 
%     a mean vector vm and space basis P and a sample x. Then the local
%     tangent coordinates are computed by
%           y = P^T * (x - vm)
%     If there are nnz non-zero entries in G, then Y has nnz columns. The 
%     output graph GM is a sparse matrix, with the values in GM telling 
%     the index of the corresponding column in Y.
%
%   - [Y, GM] = sllocaltancoords(LM, LP, X, G, ga) computes the local tangent
%     coordinates using the specified graph arrangement. There are two
%     arrangements to choose:
%       - 'N': G is m x n, models as sources, samples as targets
%              the edge connecting the i-th source and the j-th target
%              refers to the pair of the i-th model and the j-th sample
%       - 'T': G is n x m, samples as sources, models as targets
%              the edge connecting the i-th source and the j-th target
%              refers to the pair of the i-th sample and the j-th model
%
% $ Remarks $
%   - When map is not specified, the number of models should be equal to
%     the number of samples. (m = n)
%
% $ History $
%   - Created by Dahua Lin, on Sep 13rd, 2006
%

%% parse and verify input arguments

if nargin < 4
    raise_lackinput('sllocaltancoords', 4);
end

[d, m] = size(LM);
[dP, dl, m2] = size(LP);
[dX, n] = size(X);

if d ~= dP || d ~= dX
    error('sltoolbox:sizmismatch', ...
        'The sample dimensions are inconsistent among LM, LP and X');
end
if m~= m2
    error('sltoolbox:sizmismatch', ...
        'The space numbers in LM and LP are inconsistent');
end

gi = slgraphinfo(G);

if nargin < 5
    ga = 'N';
end
switch ga
    case 'N'
        if ~isequal([gi.n, gi.nt], [m, n])
            error('sltoolbox:sizmismatch', ...
                'The size of the graph does not match the samples and models');
        end
    case 'T'
        if ~isequal([gi.n, gi.nt], [n, m])
            error('sltoolbox:sizmismatch', ...
                'The size of the graph does not match the samples and models');
        end
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid graph arrangement: %s', ga);
end


%% main

% prepare indices
if ~strcmp(gi.form, 'adjmat')
    G = sladjmat(G, 'valtype', 'logical', 'sparse', true);
end
        
% prepare storage
ny = nnz(G);
Y = zeros(dl, ny);
GM = spalloc(gi.n, gi.nt, ny);

% main loop
cp = 0;
switch ga
    case 'N'
        for k = 1 : m
            vm = LM(:,k);
            P = LP(:,:,k);
            ci = find(G(k,:));
            curX = X(:, ci);
            cn = length(ci);
            curY = P' * sladdvec(curX, -vm, 1);
            Y(:, cp+1:cp+cn) = curY;
            GM(k, ci) = cp+1:cp+cn;
            cp = cp + cn;
        end
    case 'T'
        for k = 1 : m
            vm = LM(:,k);
            P = LP(:,:,k);
            ci = find(G(:,k));
            curX = X(:, ci);
            cn = length(ci);
            curY = P' * sladdvec(curX, -vm, 1);
            Y(:, cp+1:cp+cn) = curY;
            GM(ci, k) = (cp+1:cp+cn)';
            cp = cp + cn;
        end                        
end


 