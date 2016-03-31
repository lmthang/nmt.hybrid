function [unkStr, unkIndex] = findUnkStr(vocabMap)
%%
% Find a representation for unknown word in a vocabMap
% Author: Thang Luong
%%
  unkStrs = {'UUUNKKK', 'UNKNOWN', '*UNKNOWN*', 'UNK', '<UNK>', '<unk>'};
  
  %% unkStr
  unkStr = '';
  for ii=1:length(unkStrs)
    curUnkStr = unkStrs{ii};
    if isKey(vocabMap, curUnkStr)
      if strcmp(unkStr, '')
        unkStr = curUnkStr;
        break;
      end
    end
  end
  
  if strcmp(unkStr, '')
    fprintf(2, 'No vector representing unknown words\n');
    unkStr = '';
    unkIndex = -1;
  else
    assert(isKey(vocabMap, unkStr)==1);
    unkIndex = vocabMap(unkStr);
  end
end
