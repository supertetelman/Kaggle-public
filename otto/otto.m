%load reqs
pkg load statistics

%Read data in 
train = csvread('test.csv')
%test = csvread('test.csv')
%train = csvread('tiny.train.csv');
test = csvread('tiny.cv.csv');

%Initialize params
[m n] = size(test);
ksize=2;
map = zeros(ksize);

%Train kmeans centers
[idx, centers] = kmeans(train,ksize);
for i=1:ksize
	map(i) = mode(train(idx == i, 95)); %Create map the most popular classnum of each cluster to the clusternum
end

cluster = [ map centers];
csvwrite('cluster.csv', cluster)

%assign test data to clusters
[K L] = size(centers);
distance = zeros(m,K);
for i=1:K
    for j=1:L
        distance(:,i) = distance(:,i) + (test(:,j) - centers(i,j)) .^ 2;
    end
end

%TODO: this seems inefficient
[tmp idx] = min(distance');
idx = idx';

csvwrite('test_centers.csv', idx)

results = zeros(length(idx));
for i=1:length(idx)
	results(i) = map(idx(i));
end
csvwrite('results.csv', results)
