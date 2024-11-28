% set format for printing
format long
format loose

% set the seed to be 4
rng(4);

% generating the sampling points
N = 1000;
X = rand(1, N) * 2 * pi;
X = sort(X);

% generating function values at the sampled points
f = sin(X) + cos(X);

% generating the exact derivative
f_dash_exact = cos(X) - sin(X);

% numerical differentiation
f_dash_backward = ones(1, N-1);
f_dash_forward = ones(1, N-1);
f_dash_central = ones(1, N-2);

% calculating the derivatives
for i = 1:N
    
    % forward
    if i ~= N
        f_dash_forward(1, i) = (f(i+1) - f(i)) / (X(i+1) - X(i));
    end
    
    % backward
    if i ~= 1
        f_dash_backward(1, i-1) = (f(i) - f(i-1)) / (X(i) - X(i-1));
    end

    % central
    if i ~= 1 && i~=N
        f_dash_central(1, i) = (f(i+1) - f(i-1)) / (X(i+1) - X(i-1));
    end

end

% plotting

% forward difference
figure;
big_font_size = 14;
small_font_size = big_font_size - 2;
plot(X, f_dash_exact, Color='blue', LineStyle='-', LineWidth=2)
hold on
plot(X(2:N-1), f_dash_forward(1, 2:N-1), Color='red', LineStyle='--', LineWidth=4)
title("Comparison of exact derivative and forward difference for N = " + N, fontsize=big_font_size)
xlabel('x', fontsize=big_font_size)
ylabel('df/dx', fontsize=big_font_size)
legend( 'Exact', 'Forward difference', Location='best', fontsize=small_font_size)

% backward difference
figure;
plot(X, f_dash_exact, Color='blue', LineStyle='-', LineWidth=2)
hold on
plot(X(2:N-1), f_dash_backward(1, 2:N-1), Color='red', LineStyle='--', LineWidth=4)
title("Comparison of exact derivative and backward difference for N = " + N, fontsize=big_font_size)
xlabel('x', fontsize=big_font_size)
ylabel('df/dx', fontsize=big_font_size)
legend( 'Exact', 'Backward difference', Location='best', fontsize=small_font_size)

% central difference
figure;
plot(X, f_dash_exact, Color='blue', LineStyle='-', LineWidth=2)
hold on
plot(X(2:N-1), f_dash_central(1, 2:N-1), Color='red', LineStyle='--', LineWidth=4)
title("Comparison of exact derivative and central difference for N = " + N, fontsize=big_font_size)
xlabel('x', fontsize=big_font_size)
ylabel('df/dx', fontsize=big_font_size)
legend( 'Exact', 'Central difference', Location='best', fontsize=small_font_size)


% calculating root mean squared errors (mean along x)
forward_rms_error = rmse(f_dash_exact(2:N-1), f_dash_forward(2:N-1));
backward_rms_error = rmse(f_dash_exact(2:N-1), f_dash_backward(2:N-1));
central_rms_error = rmse(f_dash_exact(2:N-1), f_dash_central(2:N-1));

% printing the RMSE
disp(["Forward RMSE", "Backward RMSE", "Central RMSE"])
disp([forward_rms_error, backward_rms_error, central_rms_error])