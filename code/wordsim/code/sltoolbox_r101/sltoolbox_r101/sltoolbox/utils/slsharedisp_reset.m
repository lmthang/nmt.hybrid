function slsharedisp_reset()
%SLSHAREDISP_RESET Resets the global display options
%
% $ Syntax $
%   - slsharedisp_reset()
%
% $ History $
%   - Created by Dahua Lin, on Aug 29, 2006
%

global GLOBAL_SHARE_DISPLAYER;
global GLOBAL_SHARE_DISPLAYER_LINEBREAK;

GLOBAL_SHARE_DISPLAYER = [];
GLOBAL_SHARE_DISPLAYER_LINEBREAK = [];

