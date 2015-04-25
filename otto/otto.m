%load reqs
pkg load statistics

%Debug
debug = 0
debug_sol = 0
debug_read_sol = 0
kmeans = 1
logistic = 1
only_logistic = 1

predict = 0
do_train = 1
read_log_in = 1
read_k_in = 1

%Tunable params
lambda = 1
epsilon = .1
k_iters = 1000
log_iters = 1000
min_clusters = 23

%constants
classifiers = 9;
ksize = min_clusters



%Initialize things so they are not null depending on debug/training
map = 0; all_theta = 0; theta = 0; centers = 0;

%Read data in 
disp('Reading in training data.')
train = csvread('train.csv');
train = train(2:end,:); % remove header
train = [train (train .^ 2)];

if debug == 0
	disp('Using real dataset')
	[ mytrain cv mytest ] = makedata(train, .99, .01, .01, true);
else
	disp('Using debug dataset')
	[ mytrain cv mytest ] = makedata(train, .2, .1, .1, true);
	if debug_read_sol
		mytrain=mytrain(1:1000,:);
		cv=cv(1:100,:);
		mytest=mytest(1:100,:);
	end
	k_iters = 10
	min_clusters = 5
	log_iters = 10
end
if debug_sol == 0
	disp('Using real test set')
	test = csvread('test.csv');
	test = test(2:end,:); % remove header
	unknown = 0;
else
	disp('Using debug test set')
	test = mytest;
end

[m n] = size(test);

if  (read_log_in && ~do_train)
	disp('Reading in simple/full theta values.')
	theta = csvread([num2str(lambda) '.simple.theta.logistic.csv']);
	all_theta = csvread([num2str(lambda) '.full.theta.logistic.csv']);
end

if (read_k_in && ~do_train)
	disp('Reading in centers/kmeans map.')
	centers = csvread([num2str(ksize)  '.centers.kmeans.csv']);
	map = csvread([num2str(ksize)  '.map.kmeans.csv']);
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
		[ all_theta(:,this_theta) l_predict_train l_predict_cv l_predict_test ] = ...
			all_logistic(mytrain(predict_train == i,:), cv(predict_cv == i, :), mytest(predict_test == i,:), lambda, log_iters, map, classifiers, 'full');
	end
	csvwrite([num2str(lambda) '.full.theta.logistic.csv'],all_theta)
end

if predict
	disp('Analyzing test data and predict 1 correct value.')
	makesolution(test, theta, all_theta, centers, classifiers, map, only_logistic, kmeans, logistic)
else
	disp('Analyzing test data with % estimates.')
	makesolution2(test, theta, all_theta, centers, classifiers, map, only_logistic, kmeans, logistic)
end

%TODO:
%tune a accuracy for the clusters
%tune lambda
%tune epsilon
%Return percentages instead of prediction
%use log reg after each k
