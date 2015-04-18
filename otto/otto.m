%load reqs
pkg load statistics

%Read data in 
train = csvread('train.csv');
test = csvread('test.csv');

[ mytrain cv mytest ] = makedata(train, .6, .2, .2, true);

[ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, .99, 8, 100)

%TODO:
%tune a accuracy for the clusters
%Right function that takes centers and data and returns which centers are closest
%determin accurace vi cv rather than train
%
%Pipe each cluster into a seperate logistic regression
%tune lambda
