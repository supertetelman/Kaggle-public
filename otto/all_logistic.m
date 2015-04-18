function [ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers)

[ theta predict ]  =  runlogistic(mytrain, cv, mytest, lambda, map, classifiers, log_iters);

[m n] = size(mytrain);
predict_train = log_predict(theta,[ones(m,1) mytrain(:,2:end-1)], classifiers);
train_accuracy = sum(predict_train == mytrain(:,end))/m;

[m n] = size(cv);
predict_cv = log_predict(theta,[ones(m,1) cv(:,2:end-1)], classifiers);
cv_accuracy = sum(predict_cv == cv(:,end))/m;

[m n] = size(mytest);
predict_test = log_predict(theta,[ones(m,1) mytest(:,2:end-1)], classifiers);
test_accuracy = sum(predict_test == mytest(:,end))/m;

disp(['TRAIN RESULTS: Logistic regrestion found an accuracy of ' num2str(train_accuracy) ' percent'])
disp(['CV RESULTS: Logistic regrestion found an accuracy of ' num2str(cv_accuracy) ' percent'])
disp(['TEST RESULTS: Logistic regrestion found an accuracy of ' num2str(test_accuracy) ' percent'])

csvwrite('train.results.logistic.csv',[ mytrain(:,1) predict_train ])
csvwrite('cv.results.logistic.csv',[ cv(:,1) predict_cv ])
csvwrite('test.results.logistic.csv',[ mytest(:,1) predict_test ])
end
