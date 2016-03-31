function slsharedisp_word(varargin)
%SLSHAREDISP_WORD Displays message without line breaking
%
% $ Syntax $
%   - slsharedisp_word(...)
%
% $ History $
%   - Created by Dahua Lin, on Aug 31, 2006
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
    fprintf('%s', strmsg);
end

GLOBAL_SHARE_DISPLAYER_LINEBREAK = false; 