function S = slallsubinds(arrsiz)
%SLALLSUBINDS Generate all sub-indices for all elements of the array
%
% $ Syntax $
%   - S = slallsubinds(arrsiz)
% 
% $ Arguments $
%   - arrsiz:       the size of the array
%   - S:            the array of all sub indices
%
% $ Description $
%   - S = slallsubinds(arrsiz) generates all sub-indices for all elements
%     of the array. Suppose arrsiz is n1, n2, ..., nd. Then the output S
%     would be a d x n1 x n2 x ... x nd array, with each d-length column
%     corresponding to the sub-index of an element.
%
% $ History $
%   - Created by Dahua Lin on Jul 29th, 2006
%


arrsiz = arrsiz(:)';
d = length(arrsiz);

if d == 1
    S = (1:arrsiz)';
    
else
    totalnum = prod(arrsiz);
    ms = cumprod([1, arrsiz(1:d-1)]);
    ns = totalnum ./ (arrsiz  .* ms);
    
    S = zeros(d, totalnum);
    for i = 1 : d
        
        M = 1 : arrsiz(i);
        M = M(ones(1, ms(i)), M, ones(1, ns(i)));
        
        S(i, :) = reshape(M, [1, totalnum]);        
    end
    
    S = reshape(S, [d, arrsiz]);    
end

