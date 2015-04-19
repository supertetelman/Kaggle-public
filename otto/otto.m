%load reqs
pkg load statistics

%Debug
debug = 1
debug_sol = 1
do_train = 0
kmeans = 1
logistic = 1
only_logistic = 1
read_log_in = 1
read_k_in = 1
ksize = 9

%constants
classifiers = 9;

%Tunable params
lambda = 0
epsilon = .6
k_iters = 100
log_iters = 10
min_clusters = 9

%Initialize things so they are not null depending on debug/training
map = 0; all_theta = 0; theta = 0; centers = 0;

%Read data in 
train = csvread('train.csv');
if debug == 0
	disp('Using real dataset')
	[ mytrain cv mytest ] = makedata(train, .6, .2, .2, true);
else
	disp('Using debug dataset')
	[ mytrain cv mytest ] = makedata(train, .2, .1, .1, true);
end
if debug_sol == 0
	disp('Using real test set')
	test = csvread('test.csv');
	unknown = 0;
else
	disp('Using debug test set')
	test = mytest;
end

[m n] = size(test);

if read_log_in
	theta = csvread([num2str(lambda) '.simple.theta.logistic.csv']);
	all_theta = csvread([num2str(lambda) '.full.theta.logistic.csv']);
end

if read_k_in
	k_csv = csvread([num2str(ksize)  '.centers.kmeans.csv']);
	centers = k_csv(:,classifiers+1:end);
	map = k_csv(1,1:classifiers);
end

if (only_logistic && do_train)
	disp('Running logistic regression')
	[ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers, 'simple');
end

if (kmeans && do_train)
	disp('Running Kmeans clustering')
	[ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, epsilon, min_clusters, k_iters);
end

if (logistic && kmeans && do_train)
	disp('Running logistic regression (post Kmeans clustering)')
	all_theta = zeros(size(mytrain(:,2:end-1),2)+1,size(centers,1)* classifiers);
	for i=1:size(centers,1) %Run a sepperate regression against each cluster
		this_theta = ((i-1)*classifiers+1):(i*classifiers);
		[ all_theta(:,this_theta) predict_train predict_cv predict_test ] = ...
			all_logistic(mytrain(predict_train == i,:), cv(predict_cv == i, :), mytest(predict_test == i,:), lambda, log_iters, map, classifiers, 'full');
	end
	csvwrite([num2str(lambda) '.full.theta.logistic.csv'],all_theta)
end

theta
all_theta
c = size(theta)
c = size(all_theta)
makesolution(test, theta, all_theta, centers, classifiers, map, only_logistic, kmeans, logistic)


%TODO:
%tune a accuracy for the clusters
%tune lambda
%tune epsilon
