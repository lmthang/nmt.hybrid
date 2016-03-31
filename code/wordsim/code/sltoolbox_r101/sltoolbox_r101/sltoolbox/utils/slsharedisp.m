function slsharedisp(varargin)
%SLSHAREDISP Displays message using a shared configuration
%
% $ Syntax $
%   - slsharedisp(...)
%
% $ Description $
%   - slsharedisp(...) is to tackle the problem of sharing displaying
%     configuration across different level of functions. It uses a
%     global struct to record the displaying options, and the options
%     can be tuned by each function and passed the invoked ones. 
%     The struct is encapsulated, the user should use the following
%     function to manipulate the options.
%       - slsharedisp_attach:    attach current function to global display
%       - slsharedisp_detach:    detach current function from it
%       - slsharedisp_incindent: increase the indent of global display
%       - slsharedisp_decindent: decrease the indent of global display
%       - slsharedisp_reset:     reset the global display
%     This function is to display message using current option. 
%     The input can be either a string or follow the regulation of
%     sprintf.
%
% $ Remarks $
%   - The global display should be attached by some functions before
%     being used to display message.
%   - The action of attaching or detaching should be in pair. 
%   - If the global display is not detached due to occurrence of error,
%     it can be reset.
%   - when no function is attached to the global display, it will be
%     automatically cleared.
%   
% $ History $
%   - Created by Dahua Lin, on Aug 29, 2006
%

global GLOBAL_SHARE_DISPLAYER;
global GLOBAL_SHARE_DISPLAYER_LINEBREAK;

if isempty(GLOBAL_SHARE_DISPLAYER)
    error('sltoolbox:gdisperr', ...
        'The global displayer is not open');
end

if isempty(GLOBAL_SHARE_DISPLAYER_LINEBREAK)
    GLOBAL_SHARE_DISPLAYER_LINEBREAK = true;
end

s = GLOBAL_SHARE_DISPLAYER(end);

if s.show && ~isempty(varargin)
    nblanks = s.indent * s.indentstep;
    if length(varargin) == 1
        strmsg = varargin{1};
    else
        strmsg = sprintf(varargin{:});
    end
    if nblanks > 0 && GLOBAL_SHARE_DISPLAYER_LINEBREAK
        strmsg = [blanks(nblanks), strmsg];
    end
    disp(strmsg);
end

GLOBAL_SHARE_DISPLAYER_LINEBREAK = true;    
    

