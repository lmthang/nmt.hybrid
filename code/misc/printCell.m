function printCell(fid, cellArray, prefix)
  fprintf(fid, '%s', prefix);
  for ii=1:length(cellArray)
    if ii<length(cellArray)
      fprintf(fid, '%s ', cellArray{ii}); 
    else
      fprintf(fid, '%s\n', cellArray{ii}); 
    end
  end
end