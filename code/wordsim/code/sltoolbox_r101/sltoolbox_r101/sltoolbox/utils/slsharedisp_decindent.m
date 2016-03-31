function slsharedisp_decindent(nsteps)
%SLSHAREDISP_DECINDENT Decreases the indent of the displayer
%
% $ Syntax $
%   - slsharedisp_decindent()
%   - slsharedisp_decindent(nsteps)
%
% $ Description $
%   - slsharedisp_decindent() decreases the indent by one step.
%
%   - slsharedisp_decindent(nsteps) decreases the indent by specified
%     number of steps.
%
% $ History $
%   - Created by Dahua Lin, on Aug 29, 2006
%

global GLOBAL_SHARE_DISPLAYER;

if isempty(GLOBAL_SHARE_DISPLAYER)
    error('sltoolbox:gdisperr', ...
        'The global displayer is not open');
end

if nargin == 0
    nsteps = 1;
end

GLOBAL_SHARE_DISPLAYER(end).indent = ...
    GLOBAL_SHARE_DISPLAYER(end).indent - nsteps;