function [dc, dh, lstm_grad] = lstmUnitGrad(model, lstm, x_t, dc, dh, ll, t, srcMaxLen, zero_state, params)
  %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions.    
  if params.lstmOpt==0 % h_t = o_t * f(c_t)
    % dc = dc + f'(c_t)*o_t*dh
    % do = f'(o_t)*f(c_t)*dh
    dc = dc + params.nonlinear_f_prime(lstm{ll, t}.f_c_t).*lstm{ll, t}.o_gate.*dh;
    do = params.nonlinear_gate_f_prime(lstm{ll, t}.o_gate).*lstm{ll, t}.f_c_t .* dh;
  elseif params.lstmOpt==1 % h_t = o_t * c_t
    % dc = dc + o_t* dh
    % do = f'(o_t)*c_t*dh
    dc = dc + lstm{ll, t}.o_gate.*dh;
    do = params.nonlinear_gate_f_prime(lstm{ll, t}.o_gate).*lstm{ll, t}.c_t .* dh;
  end

  
  di = params.nonlinear_gate_f_prime(lstm{ll, t}.i_gate).*lstm{ll, t}.a_signal.*dc;
  if t>1
    df = params.nonlinear_gate_f_prime(lstm{ll, t}.f_gate).*lstm{ll, t-1}.c_t.*dc;
  else
    df = zero_state;
  end

  % arrayfun doesn't work here for GPU
  da = params.nonlinear_f_prime(lstm{ll, t}.a_signal).*lstm{ll, t}.i_gate.*dc;   

  if (t>=srcMaxLen) % grad tgt
    W = model.W_tgt{ll};
    if t==1
      lstm_grad.W_tgt = [di; df; do; da]*[x_t; zero_state]';
    else
      lstm_grad.W_tgt = [di; df; do; da]*[x_t; lstm{ll, t-1}.h_t]';
    end
  else % grad src
    W = model.W_src{ll};
    if t==1
      lstm_grad.W_src = [di; df; do; da]*[x_t; zero_state]';
    else
      lstm_grad.W_src = [di; df; do; da]*[x_t; lstm{ll, t-1}.h_t]';
    end
  end

  lstm_grad.dx = W(:, 1:params.lstmSize)'*[di; df; do; da];
  dc = lstm{ll, t}.f_gate.*dc;
  dh = W(:, params.lstmSize+1:end)'*[di; df; do; da];
  
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