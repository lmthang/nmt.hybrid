function plotTsne(inFile, labelFile, outFile, freqWordFile, varargin) % , option, isLegend
%%
% File format:
%   1st line: title
%   2nd line: y axis label
%   3rd line: headers
%   subsequent lines: data
% option: 1 -- use the first column for the x axis
% isLegend: 1 -- to print legends
%%
  colors = ['b+';'yx'; 'ro';'m*';'cs';'k^';'gx'];

  %% Argument Parser
  p = inputParser;
  % required
  addRequired(p,'inFile',@ischar);
  addRequired(p,'labelFile',@ischar);
%   addRequired(p,'option',@isnumeric);
%   addRequired(p,'isLegend',@isnumeric);
  addRequired(p,'outFile', @ischar); % number of layers
  addRequired(p,'freqWordFile', @ischar); % number of layers
  
  p.KeepUnmatched = true;
  
  parse(p,inFile, labelFile, outFile, freqWordFile, varargin{:}); % ,option,isLegend
  params = p.Results;
  
  inFile = params.inFile;
  labelFile = params.labelFile;
  freqWordFile = params.freqWordFile;
%   option = params.option;
%   isLegend = params.isLegend;
%   
%   lineWidth=6;
  
  fontsize = 18;
  % markerSize = 36;
  if strcmp(inFile, '')
    inFile = 'plotData.txt';
  end
  fprintf(1, 'inFile=%s\n', inFile);

  fid=fopen(inFile);

  % tittle
  line = fgetl(fid);
  titleStr = strtrim(line);
  fprintf(1, 'Title=%s\n', titleStr);

  % yLabel
  yLabel = fgetl(fid); %lines{lineId};
  fprintf(1, 'yLabel=%s\n', yLabel);

  % load tsne data
  [data] = dlmread(inFile);
  
  % load words
  fid = fopen(labelFile);
  textData = textscan(fid, '%s');
  words = textData{1};
  fclose(fid);
            
  % load frequent words
  if strcmp(freqWordFile, '')==0
    fid = fopen(freqWordFile);
    textData = textscan(fid, '%s');
    freqWords = textData{1};
    fclose(fid);
  else
    freqWords = {};
  end
  
  fig = figure;
  hold on;

%   if option==1 % use first column for x-axis
%     xLabel = headers{1};
%     for ii=2:numCols
%       indices = find(data(:, ii));
%       plot(data(indices, 1), data(indices, ii), [colors(ii, :) '-'], 'MarkerSize', markerSize, 'linewidth', lineWidth);
%     end
%   else
%     
%   end
  

  % ignore words
  ignoreWords = {'distrustful', 'mistrustful', 'government', 'administration', 'practicable', 'inanimate', 'unambiguous', 'unequivocal', ...
    'practical', 'penitent', 'comparing', 'compare', 'unloving', 'unloved', 'acceptance', 'advancement', 'irrelevant', 'brightness', 'colorful', ...
    'radiance', 'practicality', 'accomplished', 'accomplishment', 'development', 'nominate', 'incomprehensible', 'indispensable'};
  flags = ~ismember(words, ignoreWords);
  data = data(flags, :);
  words = words(flags);
  
  minX = min(data(:, 1));
  maxX = max(data(:, 1));
  data(:, 1) = (data(:, 1) - minX) / (maxX - minX);
  minY = min(data(:, 2));
  maxY = max(data(:, 2));
  data(:, 2) = (data(:, 2) - minY) / (maxY - minY);
  
  % check frequent words
  flags = ismember(words, freqWords);
  text(data(flags, 1), data(flags, 2), words(flags), 'Color', 'b', 'FontSize', fontsize-1); % freq
  h = text(data(~flags, 1), data(~flags, 2), words(~flags), 'Color', 'm'); % rare
  set(h, 'FontSize', fontsize);
  set(h, 'FontName', 'Times');
  set(h, 'FontAngle', 'italic');
  
  % plot(data(:, 1), data(:, 2), [colors(1, :)], 'MarkerSize', markerSize);


  %title(titleStr,'FontSize', fontsize);
  % legends
%   if option==1
%     legendStrs = headers(2:end);
%   else
%     legendStrs = headers;
%   end

%   legendLocation = 'Best'; % 'NorthWest'; % 
%   if isLegend==1 && ~isempty(legendStrs) && strcmp(legendStrs{1}, '') == 0
%     legend(legendStrs,'FontSize', fontsize, 'Location', legendLocation);
%   end

% xLabel = '';
%   fprintf(1, 'xLabel=%s\n', xLabel);
%   xlabel(xLabel,'FontSize', fontsize);
% 
%   ylabel(yLabel,'FontSize', fontsize);
%   set(gca, 'FontSize', fontsize);

  %axis tight;
  %axis([5.6, 6.8, 23, 26.5])
  
  %xlim([0, 3200])

  %axis off;
  
  % export to image
  if strcmp(params.outFile, '')==0
    print(fig, params.outFile, '-deps');
    fprintf(2, 'Saved figure to file %s\n', params.outFile);
  end
end