%load reqs
pkg load statistics

debug = 0;
debug_sol = 1;

kmeans = 1;
logistic = 1;

classifiers = 9;
lambda = 0;
epsilon = .9;
k_iters = 400;
log_iters = 500;
min_clusters = 9;


%Read data in 
train = csvread('train.csv');
if debug == 0
	[ mytrain cv mytest ] = makedata(train, .6, .2, .2, true);
else
	[ mytrain cv mytest ] = makedata(train, .2, .1, .1, true);
end
if debug_sol == 0
	test = csvread('test.csv');
else
	test = mytest;
end

[m n] = size(test);
unknown = zeros(m);

if logistic
	map = 0;
	[ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers);
	[m n] = size(test);
	solution = log_predict(theta,[ones(m,1) test(:,2:end-1)], classifiers);
	csvwrite('solution.results.logistic.csv', [test(:,1) solution ])
end

if kmeans
	[ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, epsilon, min_clusters, k_iters);

	%Create a solution set using only kmeans
	[ accuracy predict mapped ] = assess_kmeans(test, unknown, centers, map);
	k_solution = [test(:,1) mapped];
	csvwrite([num2str(length(centers)) '.solution.results.kmeans.csv'],[ k_solution ])
end

if (logistic && kmeans)
	[ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers);
	[m n] = size(test);
	solution = log_predict(theta,[ones(m,1) test(:,2:end-1)], classifiers);
	csvwrite('solution.results.logistic.csv', [test(:,1) solution ])
end
%TODO:
%tune a accuracy for the clusters
%Right function that takes centers and data and returns which centers are closest
%determin accurace vi cv rather than train
%
%Pipe each cluster into a seperate logistic regression
%tune lambda
