%load reqs
pkg load statistics

kmeans = 1;
logistic = 1;
classifiers = 9;
lambda = 0;
epsilon = .8;
k_iters = 400;
log_iters = 50;
min_clusters = 9;
debug = 1;
debug_sol = 1;


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

if kmeans
	[ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, epsilon, min_clusters, k_iters);

	%Create a solution set using only kmeans
	[ accuracy predict mapped ] = assess_kmeans(test, unknown, centers, map);
	k_solution = [test(:,1) mapped];
	csvwrite([num2str(length(centers)) '.solution.results.kmeans.csv'],[ k_solution ])
end

if logistic
	map = 0;
	[ theta predict ]  =  runlogistic(mytrain, cv, mytest, lambda, map, classifiers, log_iters);

[m n] = size(mytrain);
train_predict = log_predict(theta,[ones(m,1) mytrain(:,2:end-1)], classifiers);
train_accuracy = sum(train_predict == mytrain(:,end))/m;
[m n] = size(cv);
cv_predict = log_predict(theta,[ones(m,1) cv(:,2:end-1)], classifiers);
cv_accuracy = sum(cv_predict == cv(:,end))/m;
[m n] = size(mytest);
test_predict = log_predict(theta,[ones(m,1) mytest(:,2:end-1)], classifiers);
test_accuracy = sum(test_predict == mytest(:,end))/m;
[m n] = size(test);
solution = log_predict(theta,[ones(m,1) test(:,2:end-1)], classifiers);

disp(['TRAIN RESULTS: Logistic regrestion found an accuracy of ' num2str(train_accuracy) ' percent'])
disp(['CV RESULTS: Logistic regrestion found an accuracy of ' num2str(cv_accuracy) ' percent'])
disp(['TEST RESULTS: Logistic regrestion found an accuracy of ' num2str(test_accuracy) ' percent'])

csvwrite('train.results.logistic.csv',[ mytrain(:,1) train_predict ])
csvwrite('cv.results.logistic.csv',[ cv(:,1) cv_predict ])
csvwrite('test.results.logistic.csv',[ mytest(:,1) test_predict ])
csvwrite('solution.results.logistic.csv', [test(:,1) solution ])
end
%TODO:
%tune a accuracy for the clusters
%Right function that takes centers and data and returns which centers are closest
%determin accurace vi cv rather than train
%
%Pipe each cluster into a seperate logistic regression
%tune lambda
