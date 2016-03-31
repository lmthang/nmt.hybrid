function slsharedisp_detach()
%SLSHAREDISP_DETACH Detachs current function from global display
%
% $ Syntax $
%   - slsharedisp_detach()
%
% $ History $
%   - Created by Dahua Lin, on Aug 29, 2006
%

global GLOBAL_SHARE_DISPLAYER;

if ~isempty(GLOBAL_SHARE_DISPLAYER)
    if length(GLOBAL_SHARE_DISPLAYER) == 1
        GLOBAL_SHARE_DISPLAYER = [];
    else
        GLOBAL_SHARE_DISPLAYER(end) = [];
    end
end

        