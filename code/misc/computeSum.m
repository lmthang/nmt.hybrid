function [value] = computeSum(W, isGPU)
  value = sum(abs(W(:)));
  if isGPU
    value = gather(value);
  end
end
