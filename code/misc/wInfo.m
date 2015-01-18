function [output] = wInfo(allW)
  output = '';
  if isstruct(allW) % struct
    fields = fieldnames(allW);
    for ii=1:length(fields)
      field = fields{ii};
      if iscell(allW.(field))
        for jj=1:length(allW.(field))
          output = [output sprintf(' %s{%d}=%s', field, jj, info(allW.(field){jj}))];
        end
      else
        output = [output sprintf(' %s=%s', field, info(allW.(field)))];
      end
    end
  else % allW is not a struct
    if iscell(allW)
      for jj=1:length(allW)
        output = [output sprintf(' {%d}: %s', jj, info(allW{jj}))];
      end
    else
      output = [output sprintf(' %s', info(allW))];
    end
  end
end

function [infoStr] = info(W)
  W = full(W);
  avg = sum(sum(abs(W)))/numel(W);
  infoStr = sprintf('%.4f,%.4f,%.4f', avg, min(min(full(W))), max(max(full(W)))); %%s,, mat2str(size(W))
end
