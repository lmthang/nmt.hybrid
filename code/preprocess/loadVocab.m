function [vocab, freqs, varargout] = loadVocab(vocabFile)
%% 
% Thang Luong @ 2012, 2015 <lmthang@stanford.edu>
% Read vocab file in UTF-8 format.
%%

  fid = fopen(vocabFile,'r');
  fprintf(1, '# Loading vocab from file %s ...\n', vocabFile);
  fileLines = textscan(fid, '%s', 'delimiter', '\n'); %, 'bufsize', 100000);
  fclose(fid);
  fileLines=fileLines{1};

  vocab = cell(1,length(fileLines));
  freqs = zeros(1,length(fileLines));

  prevIndex = -1;
  for ii = 1:length(fileLines)
    tempstr = fileLines{ii};
    temp=regexp(tempstr,' ','split');
      
	  if length(temp) == 3 % word index freq
		  word = temp{1};
		  index = temp{2}; % ignore
		  freq = str2double(temp{3});
		  if prevIndex~=-1
        assert((prevIndex + 1)==index);
        prevIndex = index;
      end
    elseif length(temp) == 1 % word
      word = temp{1};
      freq = 1;
    else
      error('Invalid line format %s\n', tempstr);
    end
	  
    vocab{ii} = word;
    freqs(ii) = freq;
  end
end
