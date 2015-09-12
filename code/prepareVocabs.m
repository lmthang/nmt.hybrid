function [params] = prepareVocabs(params)
% prepareVocabs - prepare vocabs for the model
%
% Input:
%   params: parameter settings
%
% Output:
%   params: updated parameter settings with vocab fields
%
% Authors: 
%   Thang Luong @ 2015, <lmthang@stanford.edu>
%

  %% grad check
  if params.isGradCheck
    tgtVocab = {'a', 'b'};

    if params.isBi
      srcVocab = {'x', 'y'};
    end
  else
    [tgtVocab] = loadVocab(params.tgtVocabFile);    
    if params.isBi
      [srcVocab] = loadVocab(params.srcVocabFile);
    end
  end
  
  %% src vocab
  if params.isBi
    fprintf(2, '## Bilingual setting\n');
    
    % add eos, sos, zero
    srcVocab{end+1} = '<s_sos>'; % not learn
    params.srcSos = length(srcVocab);
    srcVocab{end+1} = '<s_eos>';
    params.srcEos = length(srcVocab);
    
    % here we have src eos, so we don't need tgt sos.
    params.srcVocabSize = length(srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
  end
    
  %% tgt vocab  
  params.nullPosId = 0;
    
  % add eos, sos
  tgtVocab{end+1} = '<t_sos>';
  params.tgtSos = length(tgtVocab);
  tgtVocab{end+1} = '<t_eos>';
  params.tgtEos = length(tgtVocab); 
  params.tgtVocabSize = length(tgtVocab);
  
  %% finalize vocab
  if params.isBi
    params.srcVocab = srcVocab;
  else
    %params.inVocabSize = params.tgtVocabSize;
    params.srcVocab = [];
  end
  
  params.tgtVocab = tgtVocab;
end