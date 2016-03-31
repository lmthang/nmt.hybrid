function As = slsymgraph(A, symmethod)
%SLSYMGRAPH Forces symmetry of the adjacency matrix of a graph
%
% $ Syntax $
%   - As = slsymgraph(A)
%   - As = slsymgraph(A, symmethod)
%
% $ Arguments $
%   - A:            The adjacency matrix of the original graph
%   - symmethod:    The method to symmetrize the graph
%   - As:           The symmetry adjacency matrix
% 
% $ Description $
%   - As = slsymgraph(A) makes a symmetry version of the adjacency matrix
%     A using default method.
%
%   - As = slsymgraph(A, symmethod) symmetrizes the adjacency matrix by 
%     using the specified method. 
%     \*
%     \t    Table. The method to symmetrizes adjacency matrix
%     \h       name       &              description 
%             'avgor'     & Force symmetry using the following rule:
%                           if both aij and aji are non-zeros, then 
%                           take their average
%                           if only one of aij and aji is non-zero, then
%                           take the non-zero one
%                           if both aij and aji are zeros, then set zero
%                           (for both logical and numeric)
%             'avgand'    & Force symmetry using the following rule:
%                           if both aij and aji are non-zeros, then 
%                           take their average
%                           if either one of aij and aji is zero, then
%                           set zero
%                           (for both logical and numeric)
%             'or'        & Use or-rule: d = aij | aji
%                           (for only logical)
%             'and'       & Use and-rule: d = aij & aji
%                           (for only logical)
%             'simavg'    & make simple average: always take (aij+aji)/2
%                           (for only numeric)
%     \*
%     The default method to use is 'avgor'. You can use your own function
%     handle. It should be like the following form:
%       v = f(v1, v2)
%     v1 and v2 are arrays of equal size with corresponding values. For
%     a reasonable function, it should satisfy that:
%       f(v, v) == v && f(v1, v2) = f(v2, v1)
%
% $ Remarks $
%   - A can be full matrix or sparse matrix, and As preserves the same 
%     storage form.
%
%   - A should be a square matrix.
%
%   - When 'avgor' method is applied to logical, it is equivalent to 'or'
%     when 'avgand' method is applied to logical, it is equivalent to
%     'and'.
%
% $ History $
%   - Created by Dahua Lin, on Sep 8, 2006
%

%% parse and verify input

if ndims(A) ~= 2 || size(A,1) ~= size(A,2)
    error('sltoolbox:invalidarg', ...
        'The A should be a square 2D matrix');
end
n = size(A, 1);

if nargin < 2 || isempty(symmethod)
    symmethod = 'avgor';
end

if ischar(symmethod)
    switch symmethod
        case 'avgor'
            fcs = @compsym_avgor;
        case 'avgand'
            fcs = @compsym_avgand;
        case 'or'
            fcs = @compsym_or;
            if isnumeric(A)
                error('sltoolbox:rterror', ...
                    'The or method is not applicable to numerical matrix');
            end
        case 'and'
            fcs = @compsym_and;
            if isnumeric(A)
                error('sltoolbox:rterror', ...
                    'The and method is not applicable to numerical matrix');
            end
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid method for symmetrization: %s', method);
    end
elseif isa(symmethod, 'function_handle')
    fcs = symmethod;
else
    error('sltoolbox:invalidarg', ...
        'Invalid method for symmetrization.');
end
            

%% main skeleton

% prepare all indices with aij or aji non-zero

[I0, J0] = find(A);

% single out diagonal ones
is_diag = (I0 == J0);
if any(is_diag)
    inds_diag = sub2ind([n, n], I0(is_diag), J0(is_diag));
else
    inds_diag = [];
end

% process the non-diagonal ones
not_diag = ~is_diag;
clear is_diag;

if any(not_diag)
    % filter indices    
    I0 = I0(not_diag);
    J0 = J0(not_diag);
    clear not_diag;

    % merge to down triangular part
    I = I0;
    J = J0;
    idx_ut = find(I0 > J0);
    if ~isempty(idx_ut)
        I(idx_ut) = J0(idx_ut);
        J(idx_ut) = I0(idx_ut);
    end
    clear I0 J0 idx_ut;

    % unique and expand to up triangular part
    inds_dt = sub2ind([n, n], I, J);
    [inds_dt, si] = unique(inds_dt);
    I = I(si);
    J = J(si);
    inds_ut = sub2ind([n, n], J, I);
    clear I J si;
else
    inds_dt = [];
    inds_ut = [];
end

% get original values

if ~isempty(inds_dt)
    v_dt = A(inds_dt);
    v_ut = A(inds_ut);
else
    v_dt = [];
    v_ut = [];
end

% compute the symmetrized value

v = fcs(v_dt, v_ut);
clear v_dt v_ut;

% combine diagonal value and non-diagonal value

if ~isempty(inds_diag)
    v_diag = A(inds_diag);
else
    v_diag = [];
end

s_inds = vertcat(inds_diag, inds_dt, inds_ut);
clear inds_diag inds_dt inds_ut;
s_vals = vertcat(v_diag, v, v);
clear v_diag v;

% create matrix

As = slmakeadjmat(n, n, s_inds, s_vals, islogical(A), issparse(A));
    

%% symmetry value computation functions

function vd = compsym_avgor(v1, v2)

if isnumeric(v1)        
    has_both = v1 & v2;
    only_v1 = v1 & ~v2;
    only_v2 = v2 & ~v1;
    
    vd = zeros(size(v1));
    vd(has_both) = (v1(has_both) + v2(has_both)) / 2;
    vd(only_v1) = v1(only_v1);
    vd(only_v2) = v2(only_v2);        
else
    vd = v1 | v2;            
end


function vd = compsym_avgand(v1, v2)

if isnumeric(v1)
    has_both = v1 & v2;
    
    vd = zeros(size(v1));
    vd(has_both) = (v1(has_both) + v2(has_both)) / 2;
else
    vd = v1 & v2;
end


function vd = compsym_or(v1, v2)

vd = v1 | v2;


function vd = compsym_and(v1, v2)

vd = v1 & v2;



