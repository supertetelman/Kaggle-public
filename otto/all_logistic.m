function [ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers, csv)

[ theta predict ]  =  runlogistic(mytrain, cv, mytest, lambda, map, classifiers, log_iters);

[m n] = size(mytrain);
predict_train = log_predict_sol(theta,[ones(m,1) mytrain(:,2:end-1)], classifiers);
train_accuracy = ((sum(sum((mytrain(:,end) == 1:classifiers) .* log(predict_train))))/(classifiers * m) * -1);

[m n] = size(cv);
predict_cv = log_predict_sol(theta,[ones(m,1) cv(:,2:end-1)], classifiers);
cv_accuracy = ((sum(sum((cv(:,end) == 1:classifiers) .* log(predict_cv))))/(classifiers * m) * -1);

[m n] = size(mytest);
predict_test = log_predict_sol(theta,[ones(m,1) mytest(:,2:end-1)], classifiers); %Pad this with 1s for x0
test_accuracy = ((sum(sum((mytest(:,end) == 1:classifiers) .* log(predict_test))))/(classifiers * m) * -1);

disp(['TRAIN RESULTS: Logistic regrestion found an accuracy of ' num2str(train_accuracy) ' percent'])
disp(['CV RESULTS: Logistic regrestion found an accuracy of ' num2str(cv_accuracy) ' percent'])
disp(['TEST RESULTS: Logistic regrestion found an accuracy of ' num2str(test_accuracy) ' percent'])

csvwrite([num2str(lambda) '.' csv '.theta.logistic.csv'],theta)
csvwrite([num2str(lambda) '.train.results.logistic.csv'],[ mytrain(:,1) predict_train ])
csvwrite([num2str(lambda) '.cv.results.logistic.csv'],[ cv(:,1) predict_cv ])
csvwrite([num2str(lambda) '.test.results.logistic.csv'],[ mytest(:,1) predict_test ])

size(theta)
end
