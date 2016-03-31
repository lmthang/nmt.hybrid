function logger = decindent(logger, d)
%INCINDENT Decreases the indent by a specified amount
%
% $ Syntax $
%   - logger = decindent(logger, d)
%
% $ Arguments $
%   - logger:       the target logger
%   - d:            the number of levels of indent to be decreased
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

newindent = max(logger.indent - d, 0);

logger.indent = newindent;