function [model] = loadDecodeModel(modelFile, decodeParams)
  [savedData] = load(modelFile);
  model = savedData.model;
  model.params = savedData.params;  

  % backward compatible
  model.params.attnGlobal = (model.params.attnFunc==1);
  model.params.attnLocalMono = (model.params.attnFunc==2);
  model.params.attnLocalPred = (model.params.attnFunc==4);
  if (isfield(model.params,'softmaxFeedInput')) && (~isfield(model.params,'feedInput'))
    model.params.feedInput = model.params.softmaxFeedInput;
  end
  [model.params] = backwardCompatible(model.params, {'normLocalAttn'});

  % convert absolute paths to local paths
  fieldNames = fields(model.params);
  for ii=1:length(fieldNames)
   field = fieldNames{ii};
   if ischar(model.params.(field))
     if strfind(model.params.(field), '/afs/ir/users/l/m/lmthang') ==1
       model.params.(field) = strrep(model.params.(field), '/afs/ir/users/l/m/lmthang', '~');
     end
     if strfind(model.params.(field), '/afs/cs.stanford.edu/u/lmthang') ==1
       model.params.(field) = strrep(model.params.(field), '/afs/cs.stanford.edu/u/lmthang', '~');
     end
     if strfind(model.params.(field), '/home/lmthang') ==1
       model.params.(field) = strrep(model.params.(field), '/home/lmthang', '~');
     end    
   end
  end

  % [model.params] = prepareVocabs(model.params);
  
  % copy fields
  fieldNames = fields(decodeParams);
  for ii=1:length(fieldNames)
    field = fieldNames{ii};
    if strcmp(field, 'testPrefix')==1 && strcmp(decodeParams.(field), '')==1 % skip empty testPrefix
      continue;
    elseif strcmp(field, 'testPrefix')==1
      fprintf(2, '# Decode a different test file %s\n', decodeParams.(field));
    end
    model.params.(field) = decodeParams.(field);
  end
end

%     % convert local paths to absolute paths
%     fieldNames = fields(models{mm}.params);
%     for ii=1:length(fieldNames)
%       field = fieldNames{ii};
%       if ischar(models{mm}.params.(field))
%         if strfind(models{mm}.params.(field), '~lmthang/') ==1
%           models{mm}.params.(field) = strrep(models{mm}.params.(field), '~lmthang/', '/afs/ir/users/l/m/lmthang/');
%         end
%         if strfind(models{mm}.params.(field), '~lmthang/') ==1
%           models{mm}.params.(field) = strrep(models{mm}.params.(field), '~lmthang/', '/afs/cs.stanford.edu/u/lmthang/');
%         end
%         if strfind(models{mm}.params.(field), '~lmthang/') ==1
%           models{mm}.params.(field) = strrep(models{mm}.params.(field), '~lmthang/', '/home/lmthang/');
%         end    
%       end
%     end