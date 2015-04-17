%Function that takes training data and breaks it into randomized train, cross validation, test data
%@author Adam Tetelman
%Takes a mxn data arry, 3 values 1>x>0, and a bool to randomize/not
function [ mytrain cv mytest ] = makedata(train, train_ratio, cv_ratio, test_ratio, shuffle)

[ m n ] = size(train);
train_size = floor(m * train_ratio);
cv_size =  floor(m * cv_ratio);
test_size =  floor(m * test_ratio);

if shuffle
	train = train(randperm(size(train,1)),:);
end

mytrain = train(1:train_size, :);
cv = train(train_size + 1:train_size + cv_size, :);
mytest = train(train_size + cv_size + 1:end, :);

end
