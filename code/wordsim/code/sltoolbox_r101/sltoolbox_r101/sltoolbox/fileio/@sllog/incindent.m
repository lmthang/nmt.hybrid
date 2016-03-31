function logger = incindent(logger, d)
%INCINDENT Increases the indent by a specified amount
%
% $ Syntax $
%   - logger = incindent(logger, d)
%
% $ Arguments $
%   - logger:       the target logger
%   - d:            the number of levels of indent to be increased
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

newindent = max(logger.indent + d, 0);

logger.indent = newindent;
