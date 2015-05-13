function [ theta predict_train predict_cv predict_test ] = all_logistic(mytrain, cv, mytest, lambda, log_iters, map, classifiers, csv)

[ theta predict ]  =  runlogistic(mytrain, cv, mytest, lambda, map, classifiers, log_iters);
csvwrite([num2str(lambda) '.' csv '.theta.logistic.csv'],theta)

predict_train = 0;
predict_cv = 0;
predict_test = 0

[m n] = size(mytrain);
if m > 4000
	predict_train = log_predict_sol(theta,[ones(m,1) mytrain(:,2:end-1)], classifiers);
	train_accuracy = ((sum(sum((mytrain(:,end) == 1:classifiers) .* log(predict_train))))/(classifiers * m) * -1);
	disp(['TRAIN RESULTS: Logistic regrestion found an accuracy of ' num2str(train_accuracy) ' percent'])
	csvwrite([num2str(lambda) '.train.results.logistic.csv'],[ mytrain(:,1) predict_train ])
end

[m n] = size(cv);
if m > 4000
	predict_cv = log_predict_sol(theta,[ones(m,1) cv(:,2:end-1)], classifiers);
	cv_accuracy = ((sum(sum((cv(:,end) == 1:classifiers) .* log(predict_cv))))/(classifiers * m) * -1);
	disp(['CV RESULTS: Logistic regrestion found an accuracy of ' num2str(cv_accuracy) ' percent'])
	csvwrite([num2str(lambda) '.cv.results.logistic.csv'],[ cv(:,1) predict_cv ])
end

[m n] = size(mytest);
if m > 4000
	predict_test = log_predict_sol(theta,[ones(m,1) mytest(:,2:end-1)], classifiers); %Pad this with 1s for x0
	test_accuracy = ((sum(sum((mytest(:,end) == 1:classifiers) .* log(predict_test))))/(classifiers * m) * -1);
	disp(['TEST RESULTS: Logistic regrestion found an accuracy of ' num2str(test_accuracy) ' percent'])
	csvwrite([num2str(lambda) '.test.results.logistic.csv'],[ mytest(:,1) predict_test ])
end

end
