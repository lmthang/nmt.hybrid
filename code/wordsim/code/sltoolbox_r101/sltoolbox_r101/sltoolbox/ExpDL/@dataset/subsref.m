function R = subsref(DS, ss)
%SUBSREF Get the properties by subscription
%
% $ Syntax $
%   - R = DS.<propname>
%   - R = DS.('<propname>')
%   
% $ History $
%   - Created by Dahua Lin on Jul 23, 2006
%

if ss(1).type == '.' && ischar(ss(1).subs)
    R = get(DS, ss(1).subs);
    if (length(ss) >= 2)
        R = subsref(R, ss(2:end));
    end    
else
    error('dsdml:invalidarg', 'Invalid subscription for dataset');
end


