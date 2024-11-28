% Load UCI Wine Dataset
wineData = readtable('wine/wine.data', 'FileType', 'text');
wineFeatures = wineData(:, 2:3);  
wineLabels = wineData.Var1;  
wineFeatures = normalize(wineFeatures);

% Split the data (75% train, 25% test) ensuring class balance
dataSplit = cvpartition(wineLabels, 'Holdout', 0.25);  
trainIndices = training(dataSplit);  
testIndices = test(dataSplit);  
trainFeatures = wineFeatures(trainIndices, :);  
trainLabels = wineLabels(trainIndices);  
testFeatures = wineFeatures(testIndices, :);  
testLabels = wineLabels(testIndices);  

% Pie charts for class distribution
figure;
subplot(1, 2, 1);
trainCategoricalData = categorical(trainLabels);  
orderedTrainData = reordercats(trainCategoricalData, [1, 2, 3]);  
piechart(orderedTrainData);
title('Training Data Class Distribution');

subplot(1, 2, 2);
testCategoricalData = categorical(testLabels);
orderedTestData = reordercats(testCategoricalData, [1, 2, 3]);
piechart(orderedTestData);
title('Test Data Class Distribution');

% Train classifiers
naiveBayesClassifier = fitcnb(trainFeatures, trainLabels);
ldaClassifier = fitcdiscr(trainFeatures, trainLabels);
knnClassifier = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', 5);

% Decision surfaces
N = 500;
x1Range = linspace(min(wineFeatures.Var2), max(wineFeatures.Var2), N);  
x2Range = linspace(min(wineFeatures.Var3), max(wineFeatures.Var3), N);  
[X1, X2] = meshgrid(x1Range, x2Range);
gridPoints = [X1(:), X2(:)];  

figure;
subplot(1, 3, 1);
naiveBayesPrediction = predict(naiveBayesClassifier, gridPoints);
gscatter(X1(:), X2(:), naiveBayesPrediction);
title('Naive Bayes Decision Surface');

subplot(1, 3, 2);
ldaPrediction = predict(ldaClassifier, gridPoints);
gscatter(X1(:), X2(:), ldaPrediction);
title('LDA Decision Surface');

subplot(1, 3, 3);
knnPrediction = predict(knnClassifier, gridPoints);
gscatter(X1(:), X2(:), knnPrediction);
title('KNN Decision Surface');

% Confusion Matrices
figure;
subplot(1, 3, 1);
naiveBayesTestPrediction = predict(naiveBayesClassifier, testFeatures);
confusionchart(testLabels, naiveBayesTestPrediction);
title('Naive Bayes Confusion Matrix');

subplot(1, 3, 2);
ldaTestPrediction = predict(ldaClassifier, testFeatures);
confusionchart(testLabels, ldaTestPrediction);
title('LDA Confusion Matrix');

subplot(1, 3, 3);
knnTestPrediction = predict(knnClassifier, testFeatures);
confusionchart(testLabels, knnTestPrediction);
title('KNN Confusion Matrix');

