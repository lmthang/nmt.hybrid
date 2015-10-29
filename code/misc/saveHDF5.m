function saveHDF5(fileName, model, params)
  %hdf5write(fileName, 'params', params)
  hdf5write(fileName, 'dec_wordvec', model.W_emb_tgt);
  hdf5write(fileName, 'dec_vocab', params.tgtVocab, 'WriteMode', 'append');
  
  if params.isBi
    hdf5write(fileName, 'enc_wordvec', model.W_emb_src, 'WriteMode', 'append');
    hdf5write(fileName, 'enc_vocab', params.tgtVocab, 'WriteMode', 'append');
  end
  
  % ip weight
  hdf5write(fileName, 'ip_weight', model.W_soft, 'WriteMode', 'append');
  
  % recurrent params
  input_col_ranges = 1:params.lstmSize;
  hidden_col_ranges = params.lstmSize+1:2*params.lstmSize;
  for dd=1:params.numLayers
    prefix_tgt = ['dec_d', num2str(dd), '_lstm:'];
    
    % input
    row_ranges = 1:params.lstmSize;  
    hdf5write(fileName, [prefix_tgt 'input_gate'], [model.W_tgt{dd}(row_ranges,hidden_col_ranges) model.W_tgt{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');
    
    % forget
    row_ranges = params.lstmSize+1:2*params.lstmSize;  
    hdf5write(fileName, [prefix_tgt 'forget_gate'], [model.W_tgt{dd}(row_ranges,hidden_col_ranges) model.W_tgt{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');
    
    % output
    row_ranges = 2*params.lstmSize+1:3*params.lstmSize;  
    hdf5write(fileName, [prefix_tgt 'output_gate'], [model.W_tgt{dd}(row_ranges,hidden_col_ranges) model.W_tgt{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');
    
    % signal
    row_ranges = 3*params.lstmSize+1:4*params.lstmSize;  
    hdf5write(fileName, [prefix_tgt 'input_value'], [model.W_tgt{dd}(row_ranges,hidden_col_ranges) model.W_tgt{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');
    
    if params.isBi
      prefix_src = ['enc_d', num2str(dd), '_lstm:'];
    
      % input
      row_ranges = 1:params.lstmSize;  
      hdf5write(fileName, [prefix_src 'input_gate'], [model.W_src{dd}(row_ranges,hidden_col_ranges) model.W_src{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');

      % forget
      row_ranges = params.lstmSize+1:2*params.lstmSize;  
      hdf5write(fileName, [prefix_src 'forget_gate'], [model.W_src{dd}(row_ranges,hidden_col_ranges) model.W_src{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');

      % output
      row_ranges = 2*params.lstmSize+1:3*params.lstmSize;  
      hdf5write(fileName, [prefix_src 'output_gate'], [model.W_src{dd}(row_ranges,hidden_col_ranges) model.W_src{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');

      % signal
      row_ranges = 3*params.lstmSize+1:4*params.lstmSize;  
      hdf5write(fileName, [prefix_src 'input_value'], [model.W_src{dd}(row_ranges,hidden_col_ranges) model.W_src{dd}(row_ranges, input_col_ranges)], 'WriteMode', 'append');
    end
  end
  
end