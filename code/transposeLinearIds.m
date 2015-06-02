function [transLinearIds] = transposeLinearIds(linearIds, numRows, numCols)
  rowIds = mod(linearIds, numRows);
  colIds = (linearIds-rowIds)/numRows + 1;
  transLinearIds = (rowIds-1)*numCols + colIds;
end