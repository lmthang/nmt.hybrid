function printParams(fid, params)
%%%
% Print details info about a structure. Support maps, numbers, strings, functions.
%
% Thang Luong @ 2013, <lmthang@stanford.edu>
%
%%%

  fields = fieldnames(params);
  for ii=1:length(fields)
    % map
    if isa(params.(fields{ii}), 'containers.Map')
      valueMap = params.(fields{ii});
      keyValues = keys(valueMap);
      fprintf(fid, '# %s =', fields{ii});
      for jj=1:length(keyValues)
        fprintf(fid, ' %s:%s', keyValues{jj}, valueMap(keyValues{jj}));
      end
      fprintf(fid, '\n');
    
    % numbers
    elseif isa(params.(fields{ii}), 'numeric')
      fprintf(fid, '# %s = %s\n', fields{ii}, num2str(params.(fields{ii})));
    
    % strings
    elseif isa(params.(fields{ii}), 'char')
      fprintf(fid, '# %s = %s\n', fields{ii}, params.(fields{ii}));
    
    % functions
    elseif isa(params.(fields{ii}), 'function_handle')
      f = functions(params.(fields{ii}));
      fprintf(fid, '# %s = %s\n', fields{ii}, f.function);
    end
  end
end
