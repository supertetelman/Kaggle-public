%load reqs
pkg load statistics

%Read data in 
train = csvread('train.csv')
test = csvread('test.csv')

%[ mytrain cv mytest ] = makedata(train, .6, .2, .2, false);

%%%This guy is used to have a smaller test set%%%
[ mytrain cv mytest ] = makedata(train, .05, .01, .01, true);

%Initialize params
[m n] = size(mytest);
ksize=2;
map = zeros(ksize);

%Train kmeans centers
[idx, centers] = kmeans(mytrain,ksize);
for i=1:ksize
	map(i) = mode(mytrain(idx == i, 95)); %Create map the most popular classnum of each cluster to the clusternum
end

cluster = [ map centers];
csvwrite('cluster.csv', cluster)

%assign test data to clusters
[K L] = size(centers);
distance = zeros(m,K);
for i=1:K
    for j=1:L
        distance(:,i) = distance(:,i) + (mytest(:,j) - centers(i,j)) .^ 2;
    end
end

%TODO: this seems inefficient
[tmp idx] = min(distance');
idx = idx';

csvwrite('test_centers.csv', idx)

%results = zeros(length(idx));
for i=1:length(idx)
	idx(i) = map(idx(i));
end
csvwrite('results.csv',[ mytest(:,1) idx ])
