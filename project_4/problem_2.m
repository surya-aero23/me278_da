sigma = 10;
rho = 28;
beta = 8/3;
t_end = 50;  % Total simulation time
dt = 0.01;   % Time step
steps = round(t_end / dt);

% Initial conditions
x0 = 0.9;
y0 = 0;
z0 = 0;
initial_conditions = [x0, y0, z0];

% Pre-allocate solution vectors
x = zeros(steps, 1);
y = zeros(steps, 1);
z = zeros(steps, 1);

% Initial values
x(1) = x0;
y(1) = y0;
z(1) = z0;

% Store the solution
solution = zeros(steps, 3);
solution(1, :) = initial_conditions;

% Adams-Bashforth 2nd-order method for nonlinear terms
ab2_nonlin = @(x, y, z) [sigma * (y - x); x * (rho - z) - y; x * y - beta * z];

% Time integration loop using Crank-Nicolson + AB2 method
for n = 2:steps-1
    % Compute nonlinear terms using AB2 method
    if n == 2
        % For the second time step, use Euler method for the nonlinear terms
        f_nonlin = ab2_nonlin(x(n-1), y(n-1), z(n-1));
        x(n+1) = x(n) + dt * f_nonlin(1);
        y(n+1) = y(n) + dt * f_nonlin(2);
        z(n+1) = z(n) + dt * f_nonlin(3);
    else
        % For subsequent time steps, apply the AB2 method
        f_nonlin_current = ab2_nonlin(x(n), y(n), z(n));
        f_nonlin_prev = ab2_nonlin(x(n-1), y(n-1), z(n-1));
        x(n+1) = x(n) + dt / 2 * (3 * f_nonlin_current(1) - f_nonlin_prev(1));
        y(n+1) = y(n) + dt / 2 * (3 * f_nonlin_current(2) - f_nonlin_prev(2));
        z(n+1) = z(n) + dt / 2 * (3 * f_nonlin_current(3) - f_nonlin_prev(3));
    end
    
    % Apply Crank-Nicolson method for linear terms
    % Update x using Crank-Nicolson for the linear term
    f_x = sigma * (y(n) - x(n));
    x(n+1) = x(n) + dt / 2 * (f_x + sigma * (y(n+1) - x(n+1)));
    
    % Update y using Crank-Nicolson for the linear term
    f_y = x(n) * (rho - z(n)) - y(n);
    y(n+1) = y(n) + dt / 2 * (f_y + x(n+1) * (rho - z(n+1)) - y(n+1));
    
    % Update z using Crank-Nicolson for the linear term
    f_z = x(n) * y(n) - beta * z(n);
    z(n+1) = z(n) + dt / 2 * (f_z + x(n+1) * y(n+1) - beta * z(n+1));
    
    % Store the solution
    solution(n, :) = [x(n), y(n), z(n)];
end

% Plot the results
figure;
plot3(solution(:,1), solution(:,2), solution(:,3), 'r');
xlabel('x');
ylabel('y');
zlabel('z');
title('State-space plot of Lorenz system using Crank-Nicolson + AB2');
grid on;

% Compare with ode45 results
% Define the system for ode45
lorenz_system = @(t, xyz) [sigma * (xyz(2) - xyz(1));
                            xyz(1) * (rho - xyz(3)) - xyz(2);
                            xyz(1) * xyz(2) - beta * xyz(3)];

% Solve the system using ode45
[t_ode45, sol_ode45] = ode45(lorenz_system, [0 t_end], initial_conditions);

% Plot ode45 results for comparison
figure;
plot3(sol_ode45(:,1), sol_ode45(:,2), sol_ode45(:,3), 'b');
xlabel('x');
ylabel('y');
zlabel('z');
title('State-space plot of Lorenz system using ode45');
grid on;
