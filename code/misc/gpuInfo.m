function [info] = gpuInfo(gpuDevice)
  if isstruct(gpuDevice)
    info = ['gpu, free ' num2str((gpuDevice.AvailableMemory)/2^30, '%.2fgb')];
  else
    info = '';
  end
end