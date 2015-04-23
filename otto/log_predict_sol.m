function all_predictions = log_predict_sol(theta, X, classifiers)
m = size(X, 1); % Number of training examples
all_predictions = zeros(m,classifiers); % will contain a percentage match for each classifier/row
for i=1:classifiers
	all_predictions(:,i) = sigmoid(X*theta(:,i)); %Calculate cost for all values
end

[max_prob max_index] = max(all_predictions');
predictions = max_index'; %predict the values that are the best match
