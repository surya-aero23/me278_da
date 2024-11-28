%% 2.1
% Load the dataset
dataStruct = load('Data/regression1.mat');  
featureX1 = dataStruct.x1;  
featureX2 = dataStruct.x2;  
targetY = dataStruct.y;  

% Ensure all variables are column vectors
featureX1 = featureX1(:);  
featureX2 = featureX2(:);  
targetY = targetY(:);  

% Number of data points
numDataPoints = length(targetY);  

% Set the random seed for reproducibility
rng(43);  

% Generate random indices for splitting data
randomIndices = randperm(numDataPoints);  

% Determine the split point for training and testing
trainDataSize = floor(0.8 * numDataPoints);  % 80% for training
testDataSize = numDataPoints - trainDataSize;   % 20% for testing

% Split the data into training and testing sets
trainIndices = randomIndices(1:trainDataSize);  
testIndices = randomIndices(trainDataSize+1:end);  

% Training data
trainX1 = featureX1(trainIndices);  
trainX2 = featureX2(trainIndices);  
trainTarget = targetY(trainIndices);  

% Testing data
testX1 = featureX1(testIndices);  
testX2 = featureX2(testIndices);  
testTarget = targetY(testIndices);  

% Create Polynomial Features (degree 2) for training data
trainFeatures = [ones(length(trainX1), 1), trainX1, trainX2, trainX1.^2, trainX2.^2, trainX1.*trainX2];  

% Fit Polynomial Regression Model using Least Squares
coefficients = trainFeatures \ trainTarget;  

% Make predictions on the testing data
testFeatures = [ones(length(testX1), 1), testX1, testX2, testX1.^2, testX2.^2, testX1.*testX2];  
predictedTargetTest = testFeatures * coefficients;  

% Display the coefficients
disp('Polynomial Regression Coefficients:');  
disp(coefficients);  

% Calculate the Mean Squared Error (MSE) for the test data
MSE_test = mean((predictedTargetTest - testTarget).^2);  
disp('Mean Squared Error (MSE) on Test Data:');  
disp(MSE_test);  

% Training Predictions
trainPredictedTarget = trainFeatures * coefficients;  

% Calculate the Mean Squared Error (MSE) for the training data
MSE_train = mean((trainPredictedTarget - trainTarget).^2);  
disp('Mean Squared Error (MSE) on Training Data:');  
disp(MSE_train);  

% Judging Overfitting or Underfitting
if MSE_train < MSE_test  
    disp('The model may be overfitting the data.');  
elseif MSE_train > MSE_test  
    fprintf('The model may be underfitting the data.\n');  
else  
    fprintf('The model fits the data well (no overfitting or underfitting detected).\n');  
end  

% Plotting Actual vs. Predicted values for test data
figure;  
scatter3(testX1, testX2, testTarget, 'filled');  % Actual test data points
hold on;  
scatter3(testX1, testX2, predictedTargetTest, 'r');  % Predicted test data points
title('Actual vs. Predicted Data Points');  
xlabel('Feature X1');  
ylabel('Feature X2');  
zlabel('Target Y');  
legend('Actual', 'Predicted');  

% Generate a grid for surface plotting
[x1Grid, x2Grid] = meshgrid(linspace(min(featureX1), max(featureX1), 50), linspace(min(featureX2), max(featureX2), 50));  
gridFeatures = [ones(length(x1Grid(:)), 1), x1Grid(:), x2Grid(:), x1Grid(:).^2, x2Grid(:).^2, x1Grid(:).*x2Grid(:)];  
gridPredictions = gridFeatures * coefficients;  

% Reshape predictions back into grid shape for surface plotting
gridPredictions = reshape(gridPredictions, size(x1Grid));  

% Plot the fitted surface
figure;  
surf(x1Grid, x2Grid, gridPredictions);  
hold on;  
scatter3(featureX1, featureX2, targetY, 'filled');  % Actual data points
title('Polynomial Regression Surface Fit');  
xlabel('Feature X1');  
ylabel('Feature X2');  
zlabel('Target Y');  
legend('Fitted Surface', 'Actual Data');  


%% 2.2

% Load dataset
regressionData = load('Data/regression2.mat');  
inputFeature = regressionData.x;  
targetOutput = regressionData.y;  

% Ensure column vectors
inputFeature = inputFeature(:);  
targetOutput = targetOutput(:);  

% Number of data points
numDataPoints = length(targetOutput);  

% Set random seed for reproducibility
rng(43);  
randomIndices = randperm(numDataPoints);  

% Determine training and testing split (80-20)
trainDataSize = floor(0.8 * numDataPoints);  
testDataSize = numDataPoints - trainDataSize;  
trainIndices = randomIndices(1:trainDataSize);  
testIndices = randomIndices(trainDataSize+1:end);  

% Split data into training and testing sets
trainInput = inputFeature(trainIndices);  
trainTarget = targetOutput(trainIndices);  
testInput = inputFeature(testIndices);  
testTarget = targetOutput(testIndices);  

% Create polynomial features (degree 10) for training data
polyDegree = 10;  
trainFeatures = ones(length(trainInput), polyDegree + 1);  
for i = 1:polyDegree  
    trainFeatures(:, i + 1) = trainInput.^i;  
end  

% Create polynomial features for testing data
testFeatures = ones(length(testInput), polyDegree + 1);  
for i = 1:polyDegree  
    testFeatures(:, i + 1) = testInput.^i;  
end  

% Fit polynomial regression model (unregularized)
polyCoefficients = trainFeatures \ trainTarget;  
disp('Polynomial Regression Coefficients (No Regularization):');  
disp(polyCoefficients);  

% Make predictions on test data
predictedTestOutput = testFeatures * polyCoefficients;  

% Calculate Mean Squared Error (MSE) for the test data
MSE_test = mean((predictedTestOutput - testTarget).^2);  
disp('Mean Squared Error (MSE) on Test Data (No Regularization):');  
disp(MSE_test);  

% Calculate MSE on the training data
predictedTrainOutput = trainFeatures * polyCoefficients;  
MSE_train = mean((predictedTrainOutput - trainTarget).^2);  
disp('Mean Squared Error (MSE) on Training Data (No Regularization):');  
disp(MSE_train);  

% Regularization coefficient
ridgeAlpha = 0.1;  

% Ridge Regression solution
identityMatrix = eye(polyDegree + 1);  
identityMatrix(1, 1) = 0;  % Exclude intercept from regularization
ridgeCoefficients = (trainFeatures' * trainFeatures + ridgeAlpha * identityMatrix) \ (trainFeatures' * trainTarget);  
disp('Ridge Regression Coefficients:');  
disp(ridgeCoefficients);  

% Ridge predictions on test and training data
predictedTrainOutputRidge = trainFeatures * ridgeCoefficients;  
predictedTestOutputRidge = testFeatures * ridgeCoefficients;  

% Calculate MSE for Ridge Regression
MSE_train_ridge = mean((predictedTrainOutputRidge - trainTarget).^2);  
MSE_test_ridge = mean((predictedTestOutputRidge - testTarget).^2);  
disp('Ridge Regression - Mean Squared Error (MSE) on Training Data:');  
disp(MSE_train_ridge);  
disp('Ridge Regression - Mean Squared Error (MSE) on Test Data:');  
disp(MSE_test_ridge);  

% Lasso Regression
[lassoCoefficients, lassoInfo] = lasso(trainFeatures, trainTarget, 'Lambda', ridgeAlpha);  
disp('Lasso Regression Coefficients:');  
disp(lassoCoefficients);  

% Lasso predictions on test and training data
predictedTrainOutputLasso = trainFeatures * lassoCoefficients;  
predictedTestOutputLasso = testFeatures * lassoCoefficients;  

% Calculate MSE for Lasso Regression
MSE_train_lasso = mean((predictedTrainOutputLasso - trainTarget).^2);  
MSE_test_lasso = mean((predictedTestOutputLasso - testTarget).^2);  
disp('Lasso Regression - Mean Squared Error (MSE) on Training Data:');  
disp(MSE_train_lasso);  
disp('Lasso Regression - Mean Squared Error (MSE) on Test Data:');  
disp(MSE_test_lasso);  

% Plot true vs. predicted values for polynomial regression (no regularization)
figure;  
scatter(inputFeature, targetOutput, 50, 'k', 'filled', 'DisplayName', 'True Data');  
hold on;  
% Polynomial fit for the entire dataset
fullFeatureSet = ones(length(inputFeature), polyDegree + 1);  
for i = 1:polyDegree  
    fullFeatureSet(:, i + 1) = inputFeature.^i;  
end  
fullPredictionPoly = fullFeatureSet * polyCoefficients;  
scatter(inputFeature, fullPredictionPoly, 30, 'b', 'filled', 'DisplayName', 'Polynomial Fit');  
title('Polynomial Regression - All Data');  
xlabel('Input Feature');  
ylabel('Target Output');  
legend;  
hold off;  

% Plot Ridge regression fit
figure;  
scatter(inputFeature, targetOutput, 50, 'k', 'filled', 'DisplayName', 'True Data');  
hold on;  
fullPredictionRidge = fullFeatureSet * ridgeCoefficients;  
scatter(inputFeature, fullPredictionRidge, 30, 'r', 'filled', 'DisplayName', 'Ridge Fit');  
title('Ridge Regression - All Data');  
xlabel('Input Feature');  
ylabel('Target Output');  
legend;  
hold off;  

% Plot Lasso regression fit
figure;  
scatter(inputFeature, targetOutput, 50, 'k', 'filled', 'DisplayName', 'True Data');  
hold on;  
fullPredictionLasso = fullFeatureSet * lassoCoefficients;  
scatter(inputFeature, fullPredictionLasso, 30, 'g', 'filled', 'DisplayName', 'Lasso Fit');  
title('Lasso Regression - All Data');  
xlabel('Input Feature');  
ylabel('Target Output');  
legend;  
hold off;  

% Compare all models' predictions on the entire dataset
figure;  
scatter(inputFeature, targetOutput, 50, 'k', 'filled', 'DisplayName', 'True Data');  
hold on;  
scatter(inputFeature, fullPredictionPoly, 30, 'b', 'filled', 'DisplayName', 'Polynomial');  
scatter(inputFeature, fullPredictionRidge, 30, 'r', 'filled', 'DisplayName', 'Ridge');  
scatter(inputFeature, fullPredictionLasso, 30, 'g', 'filled', 'DisplayName', 'Lasso');  
title('Comparison of All Models');  
xlabel('Input Feature');  
ylabel('Target Output');  
legend;  
hold off;  
