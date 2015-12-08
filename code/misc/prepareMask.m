function [maskInfos] = prepareMask(mask)
  T = size(mask, 2);
  % prepare mask
  maskInfos = cell(T, 1);
  for tt=1:T
    maskInfos{tt}.mask = mask(:, tt)';
    maskInfos{tt}.unmaskedIds = find(maskInfos{tt}.mask);
    maskInfos{tt}.maskedIds = find(~maskInfos{tt}.mask);
  end
end