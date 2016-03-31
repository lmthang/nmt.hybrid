function nbytes = slstructsize(S, fields, types)
%SLSTRUCTSIZE Compute the size of a struct
%
% $ Syntax $
%   - nbytes = slstructsize(S)
%   - nbytes = slstructsize(S, fields)
%   - nbytes = slstructsize(S, fields, types)
%
% $ Arguments $
%   - nbytes:            the number of bytes in the specified parts of the struct
%   - fields:            the fields that are counted in the computation
%   - types:             the types of the fields when output
%
% $ Description $
%
%   - nbytes = slstructsize(S) computes the bytes of the struct occupied
%     if S is a struct-array, then the resultant nbytes is sum of all sizes
%     of struct entries.
%
%   - nbytes = slstructsize(S, fields) only computes the specified fields.
%     If some specified fields are not in S, an error will be raised.
%
%   - nbytes = slstructsize(S, fields, types) the computation will
%     use the types for computing the size instead of using original types.
%
% $ History $
%   - Created by Dahua Lin on Dec 10th, 2005
%

%% parse and verify input arguments
if ~isstruct(S)
    error('S is not a struct');
end
if nargin < 2 || isempty(fields)
    fields = fieldnames(S);
end
if nargin < 3 || isempty(types)
    use_origin_types = true;
else
    use_origin_types = false;
end

%% compute
n = numel(S);
if n == 1
    
    nbytes = 0;
    nterms = length(fields);
    
    for t = 1 : nterms
        cur_term = S.(fields{t});
        if isempty(cur_term)
            continue;
        elseif isstruct(cur_term)
            cur_nbytes = slstructsize(cur_term);
        else
            if use_origin_types || isempty(types{t})
                cur_elem_bytes = sltypesize(class(cur_term));
            else
                cur_elem_bytes = sltypesize(types{t});
            end
            cur_nbytes = cur_elem_bytes * numel(cur_term);
        end
        
        nbytes = nbytes + cur_nbytes;
    end
    
else
    
    nbytes = 0;
    for i = 1 : n
        if use_origin_types
            nbytes = nbytes + slstructsize(S(i), fields);
        else
            nbytes = nbytes + slstructsize(S(i), fields, types);
        end
    end
    
end







