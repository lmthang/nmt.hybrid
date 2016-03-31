function varargout = sltakeval(A)
%SLTAKEVAL Extracts the values in an array/cell array to output
%
% $ Syntax $
%   -[O1, O2, ..., On] = sltakeval(A)
%
% $ Description $
%   -[O1, O2, ..., On] = sltakeval(A) takes the values in the array
%    or cell array A and assigns them to output variables.
%    The number of elements in A should be equal to the number of
%    output arguments.
%
% $ History $
%   - Created by Dahua Lin, on Sep 1st, 2006
%

n = numel(A);
if nargout ~= n
    error('sltoolbox:sizmismatch', ...
        'The number of elements in A is not equal to the number of outputs');
end



if n > 0
    if iscell(A)
        if isequal(size(A), [1, n])
            varargout = A;
        else
            varargout = reshape(A, 1, n);
        end
    else
        varargout = cell(1, n);
        for i = 1 : n
            varargout{i} = A(i);
        end
    end
end