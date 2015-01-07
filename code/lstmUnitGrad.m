function [dc, dh, lstm_grad] = lstmUnitGrad(model, lstm, dc, dh, ll, t, srcMaxLen, zero_state, params)
  if params.isGPU
    %% dh, dc
    if params.lstmOpt==0 % h_t = o_g * f(c_t)
      % dc = dc + f'(c_t)*o_g*dh
      dc = arrayfun(@plusTanhPrimeTriple, dc, lstm{ll, t}.f_c_t, lstm{ll, t}.o_gate, dh);
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % dc = dc + o_g* dh
      dc = dc + lstm{ll, t}.o_gate.*dh;
    end

    %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions. 
    % do
    if params.lstmOpt==0
      % do = f'(o_g)*f(c_t)*dh
      do = arrayfun(@sigmoidPrimeTriple, lstm{ll, t}.o_gate, lstm{ll, t}.f_c_t, dh);
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % do = f'(o_g)*c_t*dh
      do = arrayfun(@sigmoidPrimeTriple, lstm{ll, t}.o_gate, lstm{ll, t}.c_t, dh);
    end
    % di = f'(i_g) * a_signal * dc
    di = arrayfun(@sigmoidPrimeTriple, lstm{ll, t}.i_gate, lstm{ll, t}.a_signal, dc);
    
    % df = f'(f_g)*c_{t-1}*dc
    if t>1
      df = arrayfun(@sigmoidPrimeTriple, lstm{ll, t}.f_gate, lstm{ll, t-1}.c_t, dc);
    else
      df = zero_state;
    end
    % da = f'(a_signal)*i_g*dc
    da = arrayfun(@tanhPrimeTriple, lstm{ll, t}.a_signal, lstm{ll, t}.i_gate, dc);
  else
    %% dh, dc
    if params.lstmOpt==0 % h_t = o_g * f(c_t)
      % dc = dc + f'(c_t)*o_g*dh
      dc = dc + params.nonlinear_f_prime(lstm{ll, t}.f_c_t).*lstm{ll, t}.o_gate.*dh;
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % dc = dc + o_g* dh
      dc = dc + lstm{ll, t}.o_gate.*dh;
    end

    %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions. 
    % do
    if params.lstmOpt==0
      % do = f'(o_g)*f(c_t)*dh
      do = params.nonlinear_gate_f_prime(lstm{ll, t}.o_gate).*lstm{ll, t}.f_c_t .* dh;
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % do = f'(o_g)*c_t*dh
      do = params.nonlinear_gate_f_prime(lstm{ll, t}.o_gate).*lstm{ll, t}.c_t .* dh;
    end
    % di = f'(i_g) * a_signal * dc
    di = params.nonlinear_gate_f_prime(lstm{ll, t}.i_gate).*lstm{ll, t}.a_signal.*dc;
    % df = f'(f_g)*c_{t-1}*dc
    if t>1
      df = params.nonlinear_gate_f_prime(lstm{ll, t}.f_gate).*lstm{ll, t-1}.c_t.*dc;
    else
      df = zero_state;
    end
    % da = f'(a_signal)*i_g*dc
    da = params.nonlinear_f_prime(lstm{ll, t}.a_signal).*lstm{ll, t}.i_gate.*dc;   
  end
  
  % dc
  dc = lstm{ll, t}.f_gate.*dc;
  % grad W
  if (t>=srcMaxLen) % tgt
    W = model.W_tgt{ll};
  else % src
    W = model.W_src{ll};
  end
  d_ifoa = [di; df; do; da];
  lstm_grad.W = d_ifoa*lstm{ll, t}.input_xh';
  % dx, dh
  d_xh = W'*d_ifoa;
  lstm_grad.dx = d_xh(1:params.lstmSize, :); 
  dh =  d_xh(params.lstmSize+1:end, :);
  
  % clip hidden/cell derivatives
  if params.isClip
    if params.isGPU
     dh = arrayfun(@clipBackward, dh);
     dc = arrayfun(@clipBackward, dc);
    else
     dh(dh>params.clipBackward) = params.clipBackward; dh(dh<-params.clipBackward) = -params.clipBackward;
     dc(dc>params.clipBackward) = params.clipBackward; dc(dc<-params.clipBackward) = -params.clipBackward;
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

%   % mask dh, dc
%   dh = bsxfun(@times, dh, mask');
%   dc = bsxfun(@times, dc, mask');
