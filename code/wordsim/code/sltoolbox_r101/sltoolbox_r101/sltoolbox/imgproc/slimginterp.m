function V = slimginterp(A, I, J, interpker)
%SLIMGINTERP Performs image based interpolation 
%
% $ Syntax $
%   - V = slimginterp(A, I, J)
%   - V = slimginterp(A, I, J, interpker)
%
% $ Arguments $
%   - A:            The reference image array
%   - I, J:         The coordinates at which the values are interpolated
%                   The sizes of I and J should be exactly the same
%   - interpker:    The interpolation kernel: default = 'linear'.
%                   Please refer to slgetinterpkernel for details.
% 
% $ Description $
%   - V = slimginterp(A, I, J) performs interpolation on the given
%     positions specified by I and J using the default interpolator.
%     Suppose A is an array of h x w x n1 x n2 x ... nm, and X and Y
%     have size s1 x s2 x ... x sd. Then the output array V would be
%     of size s1 x s2 x ... x sd x n1 x n2 x ... nm. 
%
%   - V = slimginterp(A, I, J, interpker) performs interpolation on
%     the given positions using specified interpolator.
%
% $ History $
%   - Created by Dahua Lin, on Sep 2nd, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - use sladdvec to increase efficiency
%

%% parse and verify input arguments

if nargin < 3
    raise_lackinput('slimginterp', 3);
end

if ~isnumeric(A)
    error('sltoolbox:invalidarg', ...
        'The image array should be an numeric array');
end
dA = ndims(A);
sA = size(A);
h = sA(1); w = sA(2);

s = size(I);
if ~isequal(size(J), s)
    error('sltoolbox:invalidarg', ...
        'The sizes of I and J are inconsistent');
end

if dA == 2
    nc = 1;
else
    nc = prod(sA(3:end));
end

if nargin < 4 || isempty(interpker)
    interpker = 'linear';
end
[interpfunc, rad] = slgetinterpkernel(interpker);

%% Main skeleton

% do interpolation

if ischar(interpker) && strcmpi(interpker, 'nearest')
    V = interp_nn(A, h, w, nc, I, J, s);
else
    V = interp_kernel(A, h, w, nc, I, J, s, interpfunc, rad);
end

% reshape for multi-channel

if dA >= 4
    vsiz = [s, sA(3:end)];
    V = reshape(V, vsiz);
end

%% Core functions

function V = interp_nn(A, h, w, nc, I, J, s)

Ir = round(I);
Jr = round(J);
Ir = confine_value(Ir, 1, h);
Jr = confine_value(Jr, 1, w);
inds = ij2ind(h, Ir, Jr);
clear Ir Jr;

if nc == 1
    V = A(inds);
else
    inds = inds(:);
    A = reshape(A, h*w, nc);
    V = A(inds, :);
    V = reshape(V, [s, nc]);
end


function V = interp_kernel(A, h, w, nc, I, J, s, interpfunc, rad)

n = numel(I);
If = reshape(I, [1, n]);
Jf = reshape(J, [1, n]);

% generate indices of used points

dxs = get_offsets(rad)';
nnb = 2 * rad;
Iu = floor(If);
Ju = floor(Jf);
Iu = Iu(ones(nnb, 1), :);
Ju = Ju(ones(nnb, 1), :);
Iu = sladdvec(Iu, dxs, 1);
Ju = sladdvec(Ju, dxs, 1);

% compute displacements and weights

Di = sladdvec(Iu, -If, 2);
Dj = sladdvec(Ju, -Jf, 2);
clear If Jf;
Wi = interpfunc(Di);
clear Di;
Wj = interpfunc(Dj);
clear Dj;

% confine used indices

Iu = confine_value(Iu, 1, h);
Ju = confine_value(Ju, 1, w);

% from 1D to 2D
inds_i = expand_inds(1, nnb);
inds_j = expand_inds(2, nnb);
Wi = Wi(inds_i, :);
Wj = Wj(inds_j, :);
Iu = Iu(inds_i, :);
Ju = Ju(inds_j, :);

W = Wi .* Wj;
clear Wi Wj;
Inds = ij2ind(h, Iu, Ju);
clear Iu Ju;

% interpolation by weighted sum

if nc == 1
    M = A(Inds);
    clear Inds;
    V = sum(M .* W, 1);
    V = reshape(V, s);
else
    
% Batch implementation: the memory consumption is too large
%     Inds = Inds(:);
%     A = reshape(A, h*w, nc);
%     M = A(Inds, :);
%     clear Inds;
%     M = reshape(M, [nnb * nnb, n * nc]);
%     W = repmat(W, [1, nc]);
%     V = sum(M .* W, 1);
%     V = reshape(V, [s, nc]);

% Sequential implementation
    V = zeros(1, prod(s), nc);
    for i = 1 : nc
        curA = A(:,:,i);
        M = curA(Inds);
        curV = sum(M .* W, 1);
        V(:,:,i) = curV;
    end
    V = reshape(V, [s, nc]);

end



%% Auxiliary function

function x = confine_value(x, lb, ub)

x(x < lb) = lb;
x(x > ub) = ub;

function inds = ij2ind(h, I, J)

inds = I + h * (J - 1);

function dxs = get_offsets(r)

if r == floor(r)
    dxs = -(r-1) : r;
else
    error('sltoolbox:rterror', 'The effective radius should be integer');
end

function inds = expand_inds(d, n)

if d == 1
    inds = (1:n)';
    inds = inds(:, ones(1,n));
    inds = inds(:);
elseif d == 2
    inds = 1:n;
    inds = inds(ones(n,1), :);
    inds = inds(:);
end










