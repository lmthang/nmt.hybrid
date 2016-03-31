function slsharedisp_attach(name, varargin)
%SLSHAREDISP_ATTACH Attachs to global display options
%
% $ Syntax $
%   - slsharedisp_attach(name, ...)
%
% $ Arguments $
%   - name:     the name of the invoker
%
% $ Description $
%   - slsharedisp_attach(name, ...) attachs the current function to
%     the global display. You can specify the following properties
%       - 'show':       whether to show process information
%                       (default = true)
%       - 'indent':     the relative indent (default = 0)
%     Please note that the actual properties set to the current displayer
%     is determined by both the properties set here and the properties
%     of the previous displayer in stack:
%       new.show = current.show && previous.show;
%       new.indent = current.indent + previous.indent;     
%
% $ History $
%   - Created by Dahua Lin, on Aug 29, 2006
%

%% parse and verify input arguments

opts.show = true;
opts.indent = 0;
opts = slparseprops(opts, varargin{:});
opts.name = name;

%% Main

global GLOBAL_SHARE_DISPLAYER;

s = GLOBAL_SHARE_DISPLAYER;

if isempty(s)
    s = create_displayer([], opts);
else
    n = length(s);
    s(n+1, 1) = create_displayer(s(n), opts);
end

GLOBAL_SHARE_DISPLAYER = s;

    
    
%% Core function

function dp = create_displayer(prevdp, curdp)

dp.name = curdp.name;

if isempty(prevdp)
    dp.indent = curdp.indent;
    dp.show = curdp.show;
    dp.indentstep = 4;
else
    dp.indent = prevdp.indent + curdp.indent;
    dp.show = prevdp.show && curdp.show;
    dp.indentstep = prevdp.indentstep;
end








