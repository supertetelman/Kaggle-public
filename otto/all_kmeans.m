%This function will pass in only the raw data into all the other functions it calls, parsing out the ID and y(x) value

function [ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, epsilon, min_ksize, kiter)

%Initialize params
[m n] = size(mytest);
ksize=min_ksize;
kiter; 
occurence=epsilon;
map = zeros(ksize);

%Run with ore clusters until each cluster is consistent enough
while true
%Train and get accuracy
	map = zeros(ksize);
	%Train kmeans centers until the mode makes up X% of each cluster
	[predict_train, centers, map] = runkmeans(mytrain(:,2:end-1),  mytrain(:,end), ksize, kiter);

	accuracy = 0;
	for i=1:ksize
		cluster_i = (predict_train(:,end) == i);
		kmode = mode(mytrain(cluster_i, end));
		accurate = sum(mytrain(cluster_i, end) == kmode);
		total = sum(cluster_i);
		accuracy = accuracy + accurate/total;
	end

	disp(['TRAIN SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(accuracy/ksize) ' in the train set'])
%predict and get accuracy
	[ test_accuracy predict_test assign_test ] = assess_kmeans(mytest(:,2:end-1), mytest(:,end), centers, map);
	disp(['TEST SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(test_accuracy) ' in the cv set'])

	[ cv_accuracy predict_cv assign_cv ] = assess_kmeans(cv(:,2:end-1), cv(:,end), centers, map);
	disp(['CV SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(cv_accuracy) ' in the test set'])

	csvwrite([num2str(ksize)  '.centers.kmeans.csv' ], [ centers ])
	csvwrite([num2str(ksize)  '.map.kmeans.csv' ], [ map' ])
	csvwrite([num2str(ksize) '.train.results.kmeans.csv'],[ mytrain(:,1) predict_train ])
	csvwrite([num2str(ksize) '.test.results.kmeans.csv'],[ mytest(:,1) predict_test ])
	csvwrite([num2str(ksize) '.cv.results.kmeans.csv'],[ cv(:,1) predict_cv ])

	if accuracy/ksize > occurence
		disp(['K_MEANS TRAINING COMPLETE: Using ' num2str(ksize) ' clusters the mode classifier made up an average of ' num2str(occurence) ' % of the total values for each cluster'])
		break
	end

	ksize = ksize + 1;
end
