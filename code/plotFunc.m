function plotFunc()
  colors = ['r'; 'b';'m';'g'];
  lineWidth=2;
  markerSize=20;
  sigmas = [2.5, 5, 10];

  hold on;
  legendStrs = {};
  x=-20:0.2:20;
  for ii=1:length(sigmas)
    sigma = sigmas(ii);
    y = exp(-0.5*(x/sigma).^2);
    plot(x, y, colors(ii), 'MarkerSize', markerSize, 'linewidth', lineWidth);
    legendStrs{end+1} = ['\sigma=' num2str(sigma)];
  end

  % legend
  fontsize = 12;
  legendLocation = 'Best'; % 'NorthWest'; % 
  legend(legendStrs,'FontSize', fontsize, 'Location', legendLocation);
end
  