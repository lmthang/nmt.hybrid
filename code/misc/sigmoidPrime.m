function [result] = sigmoid_prime(z)
  result = z.*(1-z);
