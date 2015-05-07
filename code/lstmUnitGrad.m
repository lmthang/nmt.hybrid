function [lstm_grad] = lstmUnitGrad(W, lstm, c_t, c_t_1, dc, dh, ll, t, srcMaxLen, zero_state, maskedIds, params)
  dh(:, maskedIds) = 0;
  dc(:, maskedIds) = 0;
  
  if params.isGPU
    %% dh, dc
    if params.lstmOpt==0 % h_t = o_g * f(c_t)
      % dc = dc + f'(c_t)*o_g*dh
      dc = arrayfun(@plusTanhPrimeTriple, dc, lstm.f_c_t, lstm.o_gate, dh);
      % do = f'(o_g)*f(c_t)*dh
      do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, lstm.f_c_t, dh);
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % dc = dc + o_g* dh
      dc = arrayfun(@plusMult, dc, lstm.o_gate, dh);
      % do = f'(o_g)*c_t*dh
      do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, c_t, dh); % lstm.c_t
    end

    %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions. 
    % di = f'(i_g) * a_signal * dc
    di = arrayfun(@sigmoidPrimeTriple, lstm.i_gate, lstm.a_signal, dc);
    
    % df = f'(f_g)*c_{t-1}*dc
    if t>1
      df = arrayfun(@sigmoidPrimeTriple, lstm.f_gate, c_t_1, dc); % lstm{ll, t-1}.c_t
    else
      df = zero_state;
    end
    % da = f'(a_signal)*i_g*dc
    da = arrayfun(@tanhPrimeTriple, lstm.a_signal, lstm.i_gate, dc);
  else
    %% dh, dc
    if params.lstmOpt==0 % h_t = o_g * f(c_t)
      % dc = dc + f'(c_t)*o_g*dh
      dc = dc + params.nonlinear_f_prime(lstm.f_c_t).*lstm.o_gate.*dh;
      % do = f'(o_g)*f(c_t)*dh
      do = params.nonlinear_gate_f_prime(lstm.o_gate).*lstm.f_c_t .* dh;
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % dc = dc + o_g* dh
      dc = dc + lstm.o_gate.*dh;
      % do = f'(o_g)*c_t*dh
      do = params.nonlinear_gate_f_prime(lstm.o_gate).*c_t .* dh; % lstm.c_t
    end

    %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions. 
    % di = f'(i_g) * a_signal * dc
    di = params.nonlinear_gate_f_prime(lstm.i_gate).*lstm.a_signal.*dc;
    % df = f'(f_g)*c_{t-1}*dc
    if t>1
      df = params.nonlinear_gate_f_prime(lstm.f_gate).*c_t_1.*dc; % lstm{ll, t-1}.c_t
    else
      df = zero_state;
    end
    % da = f'(a_signal)*i_g*dc
    da = params.nonlinear_f_prime(lstm.a_signal).*lstm.i_gate.*dc;   
  end
  
  % dc
  lstm_grad.dc = lstm.f_gate.*dc; % contribute to grad of c_{t-1} = f_t * d(c_t) %(lstm.f_gate + lstm.f_bias)
  
  % grad W
  d_ifoa = [di; df; do; da];
  lstm_grad.W = d_ifoa*lstm.input';

  % dx, dh
  lstm_grad.input = W'*d_ifoa;
 
  % dropout
  if params.dropout<1
    if t>=srcMaxLen && ll==1 && ((params.posModel==2 && mod(t-srcMaxLen+1, 2)==0) || params.attnFunc==3 || params.attnFunc==4 || params.sameLength) % predict words
      lstm_grad.input(1:2*params.lstmSize, :) = lstm_grad.input(1:2*params.lstmSize, :).*lstm.dropoutMaskInput; % dropout x_t, s_t
    else
      lstm_grad.input(1:params.lstmSize, :) = lstm_grad.input(1:params.lstmSize, :).*lstm.dropoutMask; % dropout x_t
    end
  end
  
  % clip hidden/cell derivatives
  if params.isClip
    if params.isGPU
     lstm_grad.input = arrayfun(@clipBackward, lstm_grad.input);
     lstm_grad.dc = arrayfun(@clipBackward, lstm_grad.dc);
    else
     lstm_grad.input(lstm_grad.input>params.clipBackward) = params.clipBackward; lstm_grad.input(lstm_grad.input<-params.clipBackward) = -params.clipBackward;
     lstm_grad.dc(lstm_grad.dc>params.clipBackward) = params.clipBackward; lstm_grad.dc(lstm_grad.dc<-params.clipBackward) = -params.clipBackward;
    end
  end
end

function [clippedValue] = clipBackward(x)
  if x>1000
    clippedValue = single(1000);
  elseif x<-1000
    clippedValue = single(-1000);
  else
    clippedValue = x;
  end
end

% compute sigmoid'(x) * y * z = x * (1-x) * y * z
function [value] = sigmoidPrimeTriple(x, y, z)
  value = x*(1-x)*y*z;
end

% compute tanh'(x) * y * z = (1-x*x) * y * z
function [value] = tanhPrimeTriple(x, y, z)
  value = (1-x*x)*y*z;
end

% compute t + tanh'(x) * y * z = t + (1-x*x) * y * z
function [value] = plusTanhPrimeTriple(t, x, y, z)
  value = t + (1-x*x)*y*z;
end

% compute x + y * z
function [value] = plusMult(x, y, z)
  value = x + y*z;
end

  %lstm_grad.dx = d_xh(1:params.lstmSize, :); 
  %dh =  d_xh(params.lstmSize+1:end, :);
 
  %if params.debug==2 && params.batchId==1 && (t==srcMaxLen || t==1)
  %  fprintf(2, '# t %d, l %d\n dc:%s, dh:%s\n f_g:%s, i_g:%s, o_g:%s\n grad:%s\n', t, ll, wInfo(dc), wInfo(dh), wInfo(lstm.f_gate), wInfo(lstm.i_gate), wInfo(lstm.o_gate), wInfo(lstm_grad));
  %end
 
