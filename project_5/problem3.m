% LOAD THE DATA
data = load('Data/clustering_data.mat');
data = data.data;

% VISUALIZE THE DATA TO DETERMINE LINKAGE AND DISTANCE METRICS
figure;
scatter(data(:, 1), data(:, 2));

% Perform hierarchical clustering using average linkage
Z = linkage(data, 'average', 'euclidean');

% Plot the dendrogram
figure;
dendrogram(Z);
title('Dendrogram using Euclidean Distance and Average Linkage');
xlabel('Sample Index');
ylabel('Distance');

% Find a reasonable cutoff height for clusters (manually inspecting the dendrogram)
numClusters = 15; % This is an example, adjust based on your data

% Draw the cutoff line
hold on;
y_limits = ylim;  % Get the y-axis limits of the dendrogram plot
line([0, length(data)], [numClusters, numClusters], 'Color', 'r', 'LineStyle', '--');

% Add a label indicating the cutoff
text(length(data)*0.95, numClusters, 'Cutoff', 'Color', 'r', 'FontSize', 12, 'VerticalAlignment', 'bottom');
hold off;

% DO THE CLUSTERING

% 1. K-Means Clustering (Lloyd's Algorithm)
[idx_kmeans, C_kmeans] = kmeans(data, numClusters);

% Plot K-Means clustering results
figure;
subplot(3,1,1);
gscatter(data(:,1), data(:,2), idx_kmeans);
title('K-Means Clustering (Lloyd''s Algorithm)');
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');

% 2. Agglomerative (Hierarchical) Clustering
% Perform hierarchical clustering using average linkage
Z = linkage(data, 'average', 'euclidean');  % Euclidean distance and average linkage
idx_agglomerative = cluster(Z, 'maxclust', numClusters);

% Plot Agglomerative Clustering results
subplot(3,1,2);
gscatter(data(:,1), data(:,2), idx_agglomerative);
title('Agglomerative Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');

% 3. Gaussian Mixture Model (GMM) Clustering
% Fit a Gaussian Mixture Model with numClusters components
gmm = fitgmdist(data, numClusters);

% Assign each data point to a cluster based on the GMM
idx_gmm = cluster(gmm, data);

% Plot GMM clustering results
subplot(3,1,3);
gscatter(data(:,1), data(:,2), idx_gmm);
title('Gaussian Mixture Model (GMM) Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');


% SILHOUTTE SCORE VS #CLUSTERS
% Set range for the number of clusters
minClusters = 2;
maxClusters = 20;
avgSilhouetteScores = zeros(maxClusters - minClusters + 1, 1);

% Iterate over the number of clusters to calculate average silhouette score
for numClusters = minClusters:maxClusters
    % Perform K-Means clustering (you can replace this with Agglomerative or GMM)
    [idx, ~] = kmeans(data, numClusters);

    % Compute the silhouette scores
    silhouetteScores = silhouette(data, idx);

    % Calculate the average silhouette score
    avgSilhouetteScores(numClusters - minClusters + 1) = mean(silhouetteScores);
end

% Plot the average silhouette score vs number of clusters
figure;
plot(minClusters:maxClusters, avgSilhouetteScores, '-o', 'LineWidth', 2);
xlabel('Number of Clusters');
ylabel('Average Silhouette Score');
title('Average Silhouette Score vs Number of Clusters');
grid on;

% Identify the best number of clusters (highest average silhouette score)
[~, bestN] = max(avgSilhouetteScores);
fprintf('The best number of clusters is %d with an average silhouette score of %.4f.\n', bestN + minClusters - 1, avgSilhouetteScores(bestN));


%% SCATTER AND SILHOUTTE PLOT
% Perform K-Means Clustering (Lloyd's Algorithm)
numClusters = 6;
[idx_kmeans, C_kmeans] = kmeans(data, numClusters);

% Plot K-Means Clustering Results
figure;
subplot(3,2,1);
gscatter(data(:,1), data(:,2), idx_kmeans);
title('K-Means Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');

% Plot Silhouette for K-Means
subplot(3,2,2);
silhouette(data, idx_kmeans);
title(['K-Means Silhouette (Avg: ' num2str(mean(silhouette(data, idx_kmeans))) ')']);
xlabel('Silhouette Value');
ylabel('Cluster Index');

% Agglomerative Clustering
Z = linkage(data, 'average', 'euclidean');  % Euclidean distance and average linkage
idx_agglomerative = cluster(Z, 'maxclust', numClusters);

% Plot Agglomerative Clustering Results
subplot(3,2,3);
gscatter(data(:,1), data(:,2), idx_agglomerative);
title('Agglomerative Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');

% Plot Silhouette for Agglomerative Clustering
subplot(3,2,4);
silhouette(data, idx_agglomerative);
title(['Agglomerative Silhouette (Avg: ' num2str(mean(silhouette(data, idx_agglomerative))) ')']);
xlabel('Silhouette Value');
ylabel('Cluster Index');

% Gaussian Mixture Models (GMM)
gmm = fitgmdist(data, numClusters);  % Fit a GMM
idx_gmm = cluster(gmm, data);        % Assign clusters

% Plot GMM Clustering Results
subplot(3,2,5);
gscatter(data(:,1), data(:,2), idx_gmm);
title('GMM Clustering');
xlabel('Feature 1');
ylabel('Feature 2');
legend('show');

% Plot Silhouette for GMM
subplot(3,2,6);
silhouette(data, idx_gmm);
title(['GMM Silhouette (Avg: ' num2str(mean(silhouette(data, idx_gmm))) ')']);
xlabel('Silhouette Value');
ylabel('Cluster Index');


