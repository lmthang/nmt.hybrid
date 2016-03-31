function R = subsref(logger, ss)
%SUBSREF Get the properties by subscription
%
% $ Syntax $
%   - R = logger.<propname>
%   - R = logger.('<propname>')
%   
% $ History $
%   - Created by Dahua Lin on Aug 12nd, 2006
%


if ss(1).type == '.' && ischar(ss(1).subs)
    R = get(logger, ss(1).subs);
    if (length(ss) >= 2)
        R = subsref(R, ss(2:end));
    end    
else
    error('sltoolbox:invalidarg', 'Invalid subscription for dataset');
end