function [ theta predict ]  =  runlogistic(train, cv, test, lambda, map, classifiers, log_iters)

y = train(:,end); %Set of correct values
X = train(:,2:end-1);  % Set of data

[m n] = size(X);
X = [ones(m, 1) X]; % add X0 1s
all_theta = zeros(n+1,classifiers);

for i=1:classifiers
	initial_theta = zeros(n + 1, 1); %initialize theta

	%[cost, grad] = logisticCostFunction(initial_theta, X, y, lambda);
	options = optimset('GradObj', 'on', 'MaxIter', log_iters);
	[theta, cost] = fminunc(@(t)(logisticCostFunction(t, X, (y ==i), lambda)), initial_theta, options);
	all_theta(:,i) = theta;
end
theta = all_theta
predict = log_predict(theta, X, classifiers);

end
