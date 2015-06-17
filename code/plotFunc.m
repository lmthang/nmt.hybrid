function plotFunc(alpha)
  x=-10:0.2:10;
  y = exp(-x.^2/alpha);
  plot(x, y)
end
  