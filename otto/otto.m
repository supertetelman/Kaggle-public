%load reqs
pkg load statistics

%Debug
debug = 0
debug_sol = 1
kmeans = 1
logistic = 1
only_logistic = 1
read_log_in = 0
read_k_in = 0

%constants
classifiers = 9;

%Tunable params
lambda = 10
epsilon = .75
k_iters = 500
log_iters = 1000
min_clusters = 4


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
	theta = csvread([num2str(lambda) '.theta.logistic.csv']);
end

if read_k_in
	k_csv = csvread([num2str(ksize)  '.centers.kmeans.csv']);
	centers = k_csv(:,2:end);
	map = k_csv(:,1);
end

if only_logistic
	disp('Running logistic regression')
	map = 0;
	[ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers);
	solution = log_predict(theta,[ones(m,1) test(:,2:end-1)], classifiers);
	csvwrite('solution.results.logistic.csv', [test(:,1) solution ])
end

if kmeans
	disp('Running Kmeans clustering')
	[ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, epsilon, min_clusters, k_iters);
	[ accuracy predict mapped ] = assess_kmeans(test, unknown, centers, map);
	solution = [test(:,1) mapped];
	csvwrite([num2str(length(centers)) '.solution.results.kmeans.csv'],[ solution ])
end

if (logistic && kmeans)
	disp('Running logistic regression (post Kmeans clustering)')
	solution = [];
	for i=1:size(centers,1) %Run a sepperate regression against each cluster
		[ theta predict_train predict_cv predict_test ] = all_logistic(mytrain(predict_train == i,:), cv(predict_cv == i, :), mytest(predict_test == i,:), lambda, log_iters, map, classifiers);
		[m2 n2] = size(test(predict == i,:));
		solution = [solution; [ test(predict == i, 1)  log_predict(theta,[ones(m2,1) test(predict == i,2:end-1)], classifiers)]  ];
		csvwrite([num2str(i) '.solution.results.logistic.csv'], [solution ])
	end
end

%Format output properly
all_solution = zeros(m,classifiers+1);
all_solution(:,1) = solution(:,1);
for i=1:classifiers
	all_solution(:,i) = (solution(:,2) == i);
end
%TODO Add header
csvwrite('all.solution.results.logistic.csv', [all_solution ])

%TODO:
%tune a accuracy for the clusters
%tune lambda
%tune epsilon
