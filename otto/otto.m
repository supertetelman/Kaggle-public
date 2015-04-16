%load reqs
pkg load statistics

%Read data in 
train = csvread('test.csv')
test = csvread('test.csv')

%Train kmeans centers
ksize=2;
[idx, centers] = kmeans(train,ksize);
map = zeros(length(centers));
for i=1:length(centers)
	map(i) = mode(train(idx == i, 95)); %Create map the most popular classnum of each cluster to the clusternum
end

cluster = [ map centers];
scvwrite('cluster.csv', cluster, ',')

%assign test data to clusters
[K L] = size(centers);
distance = zeros(length(test),K);
for i=1:K
    for j=1:L
        distance(:,i) = distance(:,i) + (test(:,j) - centers(i,j)) .^ 2;
    end
end

%TODO: this seems inefficient
[tmp idx] = min(distance');
idx = idx';

scvwrite('centers.csv', idx, ',')

results = zeros(length(idx));
for i=1:length(idx)
        results(i) = map(idx(i));
end
scvwrite('results.csv', results, ',')
