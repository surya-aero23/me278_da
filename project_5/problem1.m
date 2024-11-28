% Simple Pendulum ODE Simulation and DMD Analysis

% Parameters
L = 1;                  % Length of pendulum (m)
g = 9.81;               % Acceleration due to gravity (m/s^2)
theta0 = 0.1;           % Initial angle (radians)
theta0_full = 1;        % Initial angle (radians)
theta_dot0 = 0;         % Initial angular velocity (rad/s)
tspan = 0:0.2:25;       % Time span [0, 25] with step 0.2s
params.L = L;
params.g = g;

% Initial conditions
initial_conditions = [theta0; theta_dot0];

% Solve ODE without small-angle approximation
[t, sol] = ode45(@(t, y) pendulumDynamics(t, y, params), tspan, [theta0_full; theta_dot0]);

% Solve with small-angle approximation
[t_small, sol_small] = ode45(@(t, y) pendulumSmallAngle(t, y, params), tspan, initial_conditions);

% Collect state vectors
X = sol';               % Each column is a state vector x(t) = [theta; theta_dot]
X = X(:, 1:end-25);
X_linear = sol_small';
X_linear = X_linear(:, 1:end-25);

% Create matrices X1 and X2 for DMD
X1 = X(:, 1:end-1);     % States from time step 1 to N-1
X2 = X(:, 2:end);       % States from time step 2 to N
X1_linear = X_linear(:, 1:end-1);
X2_linear = X_linear(:, 2:end);

% Calculate the DMD matrix A using Moore-Penrose pseudoinverse
A_DMD = X2 * pinv(X1);
A_linear_DMD = X2_linear * pinv(X1_linear);

% Simulate the linear system using A_DMD
x0 = [theta0_full, theta_dot0];  % Initial state
x_nonlinear_dmd = zeros(2, length(t)); % Preallocate
x_nonlinear_dmd(:,1) = x0;

x0_linear = initial_conditions;
x_linear_dmd = zeros(2, length(t)); % Preallocate
x_linear_dmd(:,1) = x0_linear;

for k = 2:length(t)
    x_nonlinear_dmd(:,k) = A_DMD * x_nonlinear_dmd(:,k-1);
    x_linear_dmd(:,k) = A_linear_DMD * x_linear_dmd(:,k-1);
end

% Function: Nonlinear Dynamics
function dydt = pendulumDynamics(~, y, params)
    theta = y(1);
    theta_dot = y(2);
    dydt = [theta_dot; - (params.g / params.L) * sin(theta)];
end

% Function: Small Angle Approximation
function dydt = pendulumSmallAngle(~, y, params)
    theta = y(1);
    theta_dot = y(2);
    dydt = [theta_dot; - (params.g / params.L) * theta];
end


% Plot comparison between original nonlinear and DMD solution
X = sol';
X_linear = sol_small';

% figure('Visible','off');
figure;

subplot(2,1,1)
plot(t, X(1,:), 'r', t, x_nonlinear_dmd(1,:), 'b--');
xlabel('Time (s)');
ylabel('Theta (rad)');
legend('Original Nonlinear Dynamics', 'DMD Nonlinear Dynamics');
title(['Angle Comparison (theta|_0 = ', num2str(theta0_full), ' rad, theta-dot|_0 = ', num2str(theta_dot0), ' rad/s)']);

subplot(2,1,2)
plot(t, X(2,:), 'r', t, x_nonlinear_dmd(2,:), 'b--');
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
legend('Original Nonlinear Dynamics', 'DMD Nonlinear Dynamics');
title('Angular Velocity Comparison');
sgtitle('Nonlinear Dynamics');

% Plot comparison between original linear and DMD solution
% figure('Visible','off');
figure;

subplot(2,1,1)
plot(t, X_linear(1,:), 'r', t, x_linear_dmd(1,:), 'b--');
xlabel('Time (s)');
ylabel('Theta (rad)');
legend('Original Linear Dynamics', 'DMD Linear Dynamics');
title(['Angle Comparison (theta|_0 = ', num2str(theta0), ' rad, theta-dot|_0 = ', num2str(theta_dot0), ' rad/s)']);

subplot(2,1,2)
plot(t, X_linear(2,:), 'r', t, x_linear_dmd(2,:), 'b--');
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
legend('Original Linear Dynamics', 'DMD Linear Dynamics');
title('Angular Velocity Comparison');
sgtitle('Linear Dynamics - Small Angle Approximation');


% A_DMD COMPARISON
% Compare A_linear_DMD with A_true
A_true = [0, 1; -g/L, 0];
exp_Atrue_delt = exp(A_true * 0.2);
error_norm_linear = norm(A_linear_DMD - exp(A_true * 0.2), 2);
error_norm_nonlinear = norm(A_DMD - exp(A_true * 0.2), 2);

fprintf('\nComparison of A_DMD with exp(A_true * dt):')
fprintf(['\nResidual norm with linear approximation: ', num2str(error_norm_linear)])
fprintf(['\nResidual norm with nonlinear dynamics: ', num2str(error_norm_nonlinear), '\n'])

% RMSE COMPARISON  
% Compute RMSE between the original nonlinear dynamics and DMD solution
theta_rmse_nonlinear = sqrt(mean((X(1,:) - x_nonlinear_dmd(1,:)).^2));
theta_dot_rmse_nonlinear = sqrt(mean((X(2,:) - x_nonlinear_dmd(2,:)).^2));

% Compute RMSE between the original linear dynamics and DMD solution
theta_rmse_linear = sqrt(mean((X_linear(1,:) - x_linear_dmd(1,:)).^2));
theta_dot_rmse_linear = sqrt(mean((X_linear(2,:) - x_linear_dmd(2,:)).^2));

% Display RMSE values
fprintf('\nRMSE Comparison for Nonlinear Dynamics:')
fprintf(['\nTheta RMSE (Nonlinear DMD): ', num2str(theta_rmse_nonlinear)])
fprintf(['\nAngular Velocity RMSE (Nonlinear DMD): ', num2str(theta_dot_rmse_nonlinear),'\n\n'])

% fprintf('\n\nRMSE Comparison for Linear Dynamics:')
% fprintf(['\nTheta RMSE (Linear DMD): ', num2str(theta_rmse_linear)])
% fprintf(['\nAngular Velocity RMSE (Linear DMD): ', num2str(theta_dot_rmse_linear), '\n'])


% COMPARING SMALL ANGLE APPROXIMATION RESULT WITH DMD RESULT
% Initial condition [1, 0] (angle = 1 rad, angular velocity = 0 rad/s)
x0_comparison = [1; 0];  

% Preallocate for linear and nonlinear dynamics
x_linear_comparison_ode45 = zeros(2, length(t)); % Linear dynamics using ode45
x_nonlinear_dmd_comparison = zeros(2, length(t)); % Nonlinear DMD

% Solve the linear system (small angle approximation) using ode45
[t_linear_ode45, sol_linear_ode45] = ode45(@(t, y) pendulumSmallAngle(t, y, params), tspan, x0_comparison);

% Solve the nonlinear dynamics using DMD (already calculated in your code)
x_linear_comparison_ode45(:, 1) = sol_linear_ode45(1, :)';  % Initial state for linear ode45
x_nonlinear_dmd_comparison(:, 1) = x0_comparison;  % Initial state for nonlinear DMD

% Simulate the linear system using ode45 and the nonlinear dynamics using DMD
for k = 2:length(t)
    % Linear system using ODE45 solution
    x_linear_comparison_ode45(:, k) = sol_linear_ode45(k, :)';
    
    % Nonlinear DMD solution
    x_nonlinear_dmd_comparison(:, k) = A_DMD * x_nonlinear_dmd_comparison(:, k-1);
end

% Plot comparison between the ODE45 solution for the linear system and the DMD solution for nonlinear dynamics
figure;

subplot(2,1,1)
plot(t, x_linear_comparison_ode45(1,:), 'r', t, x_nonlinear_dmd_comparison(1,:), 'b--');
xlabel('Time (s)');
ylabel('Theta (rad)');
legend('Linear Model (ODE45)', 'Nonlinear DMD');
title('Angle Comparison (theta|_0 = 1 rad, theta-dot|_0 = 0 rad/s)');

subplot(2,1,2)
plot(t, x_linear_comparison_ode45(2,:), 'r', t, x_nonlinear_dmd_comparison(2,:), 'b--');
xlabel('Time (s)');
ylabel('Angular Velocity (rad/s)');
legend('Linear Model (ODE45)', 'Nonlinear DMD');
title('Angular Velocity Comparison');
sgtitle('Linear Model (ODE45) vs Nonlinear DMD');
