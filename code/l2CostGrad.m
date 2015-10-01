%%%
%
% L2 cost/grad: weight*(values-refValues).^2
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
function [cost, grad] = l2CostGrad(values, refValues, weight, maskedIds, isTest)
  diff = values-refValues;
  diff(maskedIds) = 0;

  cost = weight*sum(diff.^2);
  
  % grad
  if isTest==0
    grad = 2*weight*diff;
  else
    grad = [];
  end
end
