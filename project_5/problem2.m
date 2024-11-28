A = load('Data/regression1.mat');
x1 = A.x1;
x2 = A.x2;
y = A.y;

% Ensure all variables are column vectors
x1 = x1(:);  % Make sure x1 is a column vector
x2 = x2(:);  % Make sure x2 is a column vector
y = y(:);    % Make sure y is a column vector

% Number of data points
n = length(y);

% Set the random seed for reproducibility (optional)
rng(42);

% Generate random indices for splitting
indices = randperm(n);

% Determine the split point
train_size = floor(0.8 * n);  % 80% for training
test_size = n - train_size;   % 20% for testing

% Split the data into training and testing sets
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

% Training data
x1_train = x1(train_indices);
x2_train = x2(train_indices);
y_train = y(train_indices);

% Testing data
x1_test = x1(test_indices);
x2_test = x2(test_indices);
y_test = y(test_indices);

% Create Polynomial Features for training data (degree 2)
X_train = [ones(length(x1_train), 1), x1_train, x2_train, x1_train.^2, x2_train.^2, x1_train.*x2_train];

% Fit Polynomial Regression Model using Least Squares on Training Data
b = X_train \ y_train;

% Make predictions on the testing data
X_test = [ones(length(x1_test), 1), x1_test, x2_test, x1_test.^2, x2_test.^2, x1_test.*x2_test];
y_pred = X_test * b;

% Display the coefficients
disp('Polynomial Regression Coefficients:');
disp(b);

% Calculate the Mean Squared Error (MSE) for the test data
MSE_test = mean((y_pred - y_test).^2);
disp('Mean Squared Error (MSE) on Test Data:');
disp(MSE_test);

% Training Predictions
X_train_full = [ones(length(x1_train), 1), x1_train, x2_train, x1_train.^2, x2_train.^2, x1_train.*x2_train];
y_train_pred = X_train_full * b;

% Calculate the Mean Squared Error (MSE) for the training data
MSE_train = mean((y_train_pred - y_train).^2);
disp('Mean Squared Error (MSE) on Training Data:');
disp(MSE_train);

% Judging Overfitting or Underfitting
if MSE_train < MSE_test
    disp('The model may be overfitting the data.');
elseif MSE_train > MSE_test
    disp('The model may be underfitting the data.');
else
    disp('The model is fitting the data well (no overfitting or underfitting detected).');
end

% Visualization (Optional)
% Plotting Actual vs Predicted values for test data
figure;
scatter3(x1_test, x2_test, y_test, 'filled');  % Actual data points (test)
hold on;
scatter3(x1_test, x2_test, y_pred, 'r');      % Predicted data points (test)
title('Actual vs Predicted');
xlabel('x1');
ylabel('x2');
zlabel('y');
legend('Actual', 'Predicted');

% Fitting Surface Visualization
% Generate a grid for x1 and x2 for surface plotting
[x1_grid, x2_grid] = meshgrid(linspace(min(x1), max(x1), 50), linspace(min(x2), max(x2), 50));
X_grid = [ones(length(x1_grid(:)), 1), x1_grid(:), x2_grid(:), x1_grid(:).^2, x2_grid(:).^2, x1_grid(:).*x2_grid(:)];
y_grid_pred = X_grid * b;  % Predicted values for the grid

% Reshape the predicted values back into the grid shape
y_grid_pred = reshape(y_grid_pred, size(x1_grid));

% Plot the surface
figure;
surf(x1_grid, x2_grid, y_grid_pred);
hold on;
scatter3(x1, x2, y, 'filled');  % Actual data points
title('Polynomial Regression Surface Fit');
xlabel('x1');
ylabel('x2');
zlabel('y');
legend('Fitted Surface', 'Actual Data');


%% 2.2
A = load('Data/regression2.mat');
x = A.x;
y = A.y;

x = x(:); 
y = y(:);  

n = length(y);
rng(34);
indices = randperm(n);

train_size = floor(0.8 * n);
test_size = n - train_size;
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

x_train = x(train_indices);
y_train = y(train_indices);
x_test = x(test_indices);
y_test = y(test_indices);

degree = 10;  % Degree of the polynomial
X_train = ones(length(x_train), degree+1);  % Initialize X_train matrix
for i = 1:degree
    X_train(:,i+1) = x_train.^i;  % Create powers of x for each degree
end

X_test = ones(length(x_test), degree+1);  % Initialize X_test matrix
for i = 1:degree
    X_test(:,i+1) = x_test.^i;  % Create powers of x for each degree
end

b = X_train \ y_train;

disp('Polynomial Regression Coefficients (No Regularization):');
disp(b);

y_pred = X_test * b;

% Calculate the Mean Squared Error (MSE) for the test data
MSE_test = mean((y_pred - y_test).^2);
disp('Mean Squared Error (MSE) on Test Data (No Regularization):');
disp(MSE_test);

% Calculate the MSE on the training data
y_train_pred = X_train * b;  % Predictions on the training data
MSE_train = mean((y_train_pred - y_train).^2);
disp('Mean Squared Error (MSE) on Training Data (No Regularization):');
disp(MSE_train);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Regularization coefficient
alpha = 0.1;

% Ridge Regression
% Create the Ridge regression coefficient matrix
I = eye(degree + 1);  % Identity matrix
I(1, 1) = 0;  % Don't regularize the intercept term

% Compute the Ridge solution
b_ridge = (X_train' * X_train + alpha * I) \ (X_train' * y_train);

disp('Ridge Regression Coefficients:');
disp(b_ridge);

% Predictions on Test and Training Data
y_train_pred_ridge = X_train * b_ridge;  % Training data predictions
y_pred_ridge = X_test * b_ridge;  % Test data predictions

% Calculate MSE for Ridge Regression
MSE_train_ridge = mean((y_train_pred_ridge - y_train).^2);
MSE_test_ridge = mean((y_pred_ridge - y_test).^2);

disp('Ridge Regression - Mean Squared Error (MSE) on Training Data:');
disp(MSE_train_ridge);
disp('Ridge Regression - Mean Squared Error (MSE) on Test Data:');
disp(MSE_test_ridge);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Lasso Regression
% Use lasso function to compute the Lasso coefficients
[b_lasso, FitInfo] = lasso(X_train, y_train, 'Lambda', alpha);

disp('Lasso Regression Coefficients:');
disp(b_lasso);

% Predictions on Test and Training Data
y_train_pred_lasso = X_train * b_lasso;  % Training data predictions
y_pred_lasso = X_test * b_lasso;  % Test data predictions

% Calculate MSE for Lasso Regression
MSE_train_lasso = mean((y_train_pred_lasso - y_train).^2);
MSE_test_lasso = mean((y_pred_lasso - y_test).^2);

disp('Lasso Regression - Mean Squared Error (MSE) on Training Data:');
disp(MSE_train_lasso);
disp('Lasso Regression - Mean Squared Error (MSE) on Test Data:');
disp(MSE_test_lasso);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create a figure for the plots
figure;

% Plot the true data for the entire dataset
scatter(x, y, 50, 'k', 'filled', 'DisplayName', 'True Data');  % '50' for marker size, 'k' for black
hold on;

% Scatter plot for the unregularized (Polynomial) fit on the entire dataset
X_all = ones(length(x), degree + 1);  % Include all data points (train + test)
for i = 1:degree
    X_all(:, i + 1) = x.^i;  % Create powers of x for each degree
end
y_pred_poly_all = X_all * b;  % Polynomial predictions for all data points
scatter(x, y_pred_poly_all, 30, 'b', 'filled', 'DisplayName', 'Polynomial Fit (No Regularization)');

title('Polynomial Regression (No Regularization) - All Data');
xlabel('x');
ylabel('y');
legend;
hold off;

% Plot the Ridge regularized fit for the entire dataset
figure;
scatter(x, y, 50, 'k', 'filled', 'DisplayName', 'True Data');
hold on;

y_pred_ridge_all = X_all * b_ridge;  % Ridge predictions for all data points
scatter(x, y_pred_ridge_all, 30, 'r', 'filled', 'DisplayName', 'Ridge Fit');

title('Ridge Regression - All Data');
xlabel('x');
ylabel('y');
legend;
hold off;

% Plot the Lasso regularized fit for the entire dataset
figure;
scatter(x, y, 50, 'k', 'filled', 'DisplayName', 'True Data');
hold on;

y_pred_lasso_all = X_all * b_lasso;  % Lasso predictions for all data points
scatter(x, y_pred_lasso_all, 30, 'g', 'filled', 'DisplayName', 'Lasso Fit');

title('Lasso Regression - All Data');
xlabel('x');
ylabel('y');
legend;
hold off;

% Comparison of all models' predictions on the entire dataset
figure;
scatter(x, y, 50, 'k', 'filled', 'DisplayName', 'True Data');
hold on;

% Scatter for all fits with different colors and markers
scatter(x, y_pred_poly_all, 30, 'b', 'filled', 'DisplayName', 'Polynomial (No Regularization)');
scatter(x, y_pred_ridge_all, 30, 'r', 'filled', 'DisplayName', 'Ridge');
scatter(x, y_pred_lasso_all, 30, 'g', 'filled', 'DisplayName', 'Lasso');

title('Comparison of All Models on All Data');
xlabel('x');
ylabel('y');
legend;
hold off;