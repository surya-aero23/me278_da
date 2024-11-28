function Q4

% Load UCI Wine Dataset
data = readtable('wine/wine.data', 'FileType', 'text');
features = data(:, 2:3); % Alcohol and Malic Acid
labels = data.Var1; % Wine classes

% Normalize features
features = normalize(features);


% Split the data (75% train, 25% test) ensuring class balance
cv = cvpartition(labels, 'Holdout', 0.25);
trainIdx = training(cv);
testIdx = test(cv);

X_train = features(trainIdx, :);
y_train = labels(trainIdx);
X_test = features(testIdx, :);
y_test = labels(testIdx);

% Pie charts for class distribution
figure;
subplot(1, 2, 1);
pie(histcounts(categorical(y_train)));
title('Training Data Class Distribution');

subplot(1, 2, 2);
%disp(y_test)
pie(histcounts(categorical(y_test)));
title('Test Data Class Distribution');

% Train classifiers
% 1. Naive Bayes
nbModel = fitcnb(X_train, y_train);

% 2. Discriminant Analysis
ldaModel = fitcdiscr(X_train, y_train);

% 3. K-Nearest Neighbors (k=5)
knnModel = fitcknn(X_train, y_train, 'NumNeighbors', 5);

% Plot decision surfaces
x1_range = linspace(min(features.Var2), max(features.Var2), 100);
x2_range = linspace(min(features.Var3), max(features.Var3), 100);
[X1, X2] = meshgrid(x1_range, x2_range);
grid = [X1(:), X2(:)];

figure;
% Naive Bayes
subplot(1, 3, 1);
nbPred = predict(nbModel, grid);
gscatter(X1(:), X2(:), nbPred);
title('Naive Bayes Decision Surface');

% Discriminant Analysis
subplot(1, 3, 2);
ldaPred = predict(ldaModel, grid);
gscatter(X1(:), X2(:), ldaPred);
title('LDA Decision Surface');

% KNN
subplot(1, 3, 3);
knnPred = predict(knnModel, grid);
gscatter(X1(:), X2(:), knnPred);
title('KNN Decision Surface');

% Confusion Matrices
figure;
% Naive Bayes
subplot(1, 3, 1);
nbTestPred = predict(nbModel, X_test);
confusionchart(y_test, nbTestPred);
title('Naive Bayes Confusion Matrix');

% Discriminant Analysis
subplot(1, 3, 2);
ldaTestPred = predict(ldaModel, X_test);
confusionchart(y_test, ldaTestPred);
title('LDA Confusion Matrix');

% KNN
subplot(1, 3, 3);
knnTestPred = predict(knnModel, X_test);
confusionchart(y_test, knnTestPred);
title('KNN Confusion Matrix');


end
