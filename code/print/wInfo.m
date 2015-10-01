function [output] = wInfo(allW, opt)

  if ~exist('opt', 'var')
    opt = -1;
  end
  
  output = '';
  if isstruct(allW) % struct
    fields = fieldnames(allW);
    for ii=1:length(fields)
      field = fields{ii};
      if iscell(allW.(field))
        for jj=1:length(allW.(field))
          output = [output sprintf(' %s{%d}=%s', field, jj, info(allW.(field){jj}, opt))];
        end
      else
        output = [output sprintf(' %s=%s', field, info(allW.(field), opt))];
      end
    end
  else % allW is not a struct
    if iscell(allW)
      for jj=1:length(allW)
        output = [output sprintf(' {%d}: %s', jj, info(allW{jj}, opt))];
      end
    else
      output = [output sprintf(' %s', info(allW, opt))];
    end
  end
end

function [infoStr] = info(W, opt)
  if opt==0 % integer
    infoStr = sprintf('%d', W);
  elseif opt==1 % float
    infoStr = sprintf('%.2f', W);
  else
    avg = sum(sum(abs(W)))/numel(W);
    infoStr = sprintf('%.3f', avg);
  end
  %infoStr = sprintf('%.3f,%.3f,%.3f', avg, min(min(full(W))), max(max(full(W)))); 
  %infoStr = sprintf('%s,%.3f,%.3f,%.3f', mat2str(size(W)), avg, min(min(full(W))), max(max(full(W)))); 
end
