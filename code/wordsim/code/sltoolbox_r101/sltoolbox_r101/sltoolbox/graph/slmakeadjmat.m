function A = slmakeadjmat(n, nt, edges, vals, islogic, isspar)
%SLMAKEADJMAT Makes an adjacency matrix using edges and corresponing values
%
% $ Syntax $
%   - A = slmakeadjmat(n, nt, edges, vals, islogic, issparse)
%
% $ Arguments $
%   - n:        The number of (source) nodes
%   - nt:       The number of (target) nodes
%   - edges:    The set of edges 
%   - vals:     The values associated with edges
%   - islogic:  whether to make a logical matrix
%   - isspar:   whether to make a sparse matrix
%   - A:        The constructed adjacency matrix
%
% $ Description $
%   - A = slmakeadjmat(n, nt, edges, vals, islogic, isspar) makes an 
%     adjacency matrix using edges and corresponing values. Here, edges 
%     and vals have the following configurations, and the value type of 
%     the output matrix is determined by the configuration and the type 
%     of vals.
%       1. edges has 1 or 2 columns, vals is empty: a matrix with 
%          all elements corresponding to existent edges set to 1.
%       2. edges has 3 columns, vals is empty: a matrix with 
%          all elements set using the 3rd column of edges.
%       3. edges has 1 or 2 or 3 columns, vals has 1 column: a matrix 
%          the values in vals are set to the matrix. The value column 
%          in edges is ignored.
%     When edges has 1 column, it contains the linear index of the edge 
%     elements, when it has 2 columns, it contains the I and J subscripts
%     of the edge elements, when it has 3 columns, it contains the I and J
%     subscripts and the values.
%  
% $ Remarks $
%   - It is an internal support function, no checking will be performed.
%
% $ History $
%   - Created by Dahua Lin, on Sep 9, 2006
%

%% Main skeleton

if isempty(edges)
    A = make_emptymat(n, nt, 0, islogic, isspar);        
else
    [ne, ncols] = size(edges);
    
    % decide indices
    inds = [];
    if isspar
        if ncols == 1
            inds = edges;
        else
            I = edges(:,1);
            J = edges(:,2);
        end
    else
        if ncols == 1
            inds = edges;
        else
            inds = sub2ind([n, nt], edges(:,1), edges(:,2));
        end
    end
    
    % decide values
    if isempty(vals)
        if ncols == 1 || ncols == 2
            if islogic
                vals = true;
            else
                vals = 1;
            end
        else
            if islogic
                vals = (edges(:,3) ~= 0);
            else
                vals = edges(:,3);
            end
        end
    else
        if islogic
            if ~islogical(vals)
                vals = (vals ~= 0);
            end
        else
            if ~isa(vals, 'double')
                vals = double(vals);
            end
        end
    end
                
    
    % do construction
    if isempty(inds)    % use I and J to construct sparse matrix        
        A = sparse(I, J, vals, n, nt);
    else                % create empty matrix first then use linear indices to fill
        A = make_emptymat(n, nt, ne, islogic, isspar);
        A(inds) = vals;
    end                                          
    
end


%% Auxiliary functions

function A = make_emptymat(n, nt, ne, islogic, isspar)

if isspar
    if islogic
        if ne > 0
            A = sparse(1, 1, false, n, nt, ne);
        else
            A = sparse(1, 1, false, n, nt);
        end
    else
        A = spalloc(n, nt, ne);
    end
else
    if islogic
        A = false(n, nt);
    else
        A = zeros(n, nt);
    end
end
            


    



