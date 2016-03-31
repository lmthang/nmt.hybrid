function logger = close(logger)
%CLOSE Closes the logger
%
% $ Syntax $
%   - logger = close(logger)
%
% $ Description $
%   - logger = close(logger) closes the logger by closing all filehandles
%     and set the winshow to false
%
% $ History $
%   - Created by Dahua Lin, on Aug 12nd, 2006
%

logger.winshow = false;

if ~isempty(logger.files)
    nf = length(logger.files);
    F = logger.files;
    
    for i = 1 : nf
        fclose(F(i).fid);
    end
    
    logger.files = [];
end
