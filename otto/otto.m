%load reqs
pkg load statistics

classifiers = 9;

%Read data in 
train = csvread('train.csv');
%[ mytrain cv mytest ] = makedata(train, .6, .2, .2, true);
[ mytrain cv mytest ] = makedata(train, .2, .1, .1, true);
test = mytest;

%test = csvread('test.csv');
[m n] = size(test);
unknown = zeros(m);


[ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, .65, 8, 100);

%Create a solution set using only kmeans
[ accuracy predict mapped ] = assess_kmeans(test, unknown, centers, map);
k_solution = [test(:,1) mapped];
csvwrite([num2str(length(centers)) '.solution.results.kmeans.csv'],[ k_solution ])



%TODO:
%tune a accuracy for the clusters
%Right function that takes centers and data and returns which centers are closest
%determin accurace vi cv rather than train
%
%Pipe each cluster into a seperate logistic regression
%tune lambda
