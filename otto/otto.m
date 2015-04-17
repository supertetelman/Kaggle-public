%load reqs
pkg load statistics

%Read data in 
train = csvread('train.csv');
test = csvread('test.csv');

%[ mytrain cv mytest ] = makedata(train, .6, .2, .2, false);

%%%This guy is used to have a smaller test set%%%
[ mytrain cv mytest ] = makedata(train, .005, .01, .01, true);

%Initialize params
[m n] = size(mytest);
ksize=2;
kiter=100;
occurence=.95
map = zeros(ksize);


%Run with ore clusters until each cluster is consistent enough
while true
	map = zeros(ksize);
	%Train kmeans centers until the mode makes up X% of each cluster
	[predict, centers, map] = runkmeans(mytrain,ksize);
	%mytrain = [ mytrain idx ]; %add predictions tdo training data set
	csvwrite([num2str(ksize)  'cluster.csv' ], [ map' centers ])

	accuracy = 0
	for i=1:ksize
		cluster_i = (predict(:,end) == i);
		kmode = mode(mytrain(cluster_i, end));
		accurate = sum(mytrain(cluster_i, end) == kmode);
		total = sum(cluster_i);
		accuracy = accuracy + accurate/total;
	end

	disp(['With ' num2str(ksize) ' cluster we had an accuracy of ' num2str(accuracy/ksize) ])
	if accuracy/ksize > occurence
		disp(['Using ' num2str(ksize) 'clusters'])
		break
	end
	ksize = ksize + 1;
end

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
m=length(idx)
for i=1:m
	idx(i) = map(idx(i));
end
csvwrite('results.csv',[ mytest(:,1) idx ])
accuracy = sum(idx == mytest(:,end))/m;
disp(['TEST SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(accuracy) ' in the test set'])



%TODO:
%tune a accuracy for the clusters
%Right function that takes centers and data and returns which centers are closest
%determin accurace vi cv rather than train
%
%Pipe each cluster into a seperate logistic regression
%tune lambda

