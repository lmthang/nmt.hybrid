function cprops = edl_check_internalindices(props)
%EDL_CHECK_INTERNALINDICES Checks the consistency of internal indices
% 
% $ Syntax $
%   - cprops = edl_check_internalindices(props)
%
% $ Arguments $
%   - props:        the struct array of property entries
%   - cprops:       the converted array with all indices coverted to
%                   numeric
%
% $ History $
%   - Created by Dahua Lin, on Aug 14, 2006
%


cprops = props;

if ~isempty(props)
    
    n = length(props);

    if ~isfield(props, 'internal_index')
        error('edl:parseerror', ...
            'The entries do not have the required field: internal_index');
    end

    for i = 1 : n
        curidx = str2double(props(i).internal_index);
        if curidx ~= i
            error('edl:parseerror', ...
                'Internal index inconsistency');
        end
        cprops(i).internal_index = curidx;
    end

end