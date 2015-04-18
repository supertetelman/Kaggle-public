function [ map centers predict_train predict_cv predict_test] = all_kmeans(mytrain, cv, mytest, epsilon, min_ksize, kiter)

%Initialize params
[m n] = size(mytest);
ksize=min_ksize;
kiter; %TODO
occurence=epsilon
map = zeros(ksize);


%Run with ore clusters until each cluster is consistent enough
while true
%Train and get accuracy
	map = zeros(ksize);
	%Train kmeans centers until the mode makes up X% of each cluster
	[predict_train, centers, map] = runkmeans(mytrain,ksize);

	accuracy = 0;
	for i=1:ksize
		cluster_i = (predict_train(:,end) == i);
		kmode = mode(mytrain(cluster_i, end));
		accurate = sum(mytrain(cluster_i, end) == kmode);
		total = sum(cluster_i);
		accuracy = accuracy + accurate/total;
	end

	disp(['TRAIN SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(accuracy/ksize) ' in the test set'])
	if accuracy/ksize > occurence
		disp(['Using ' num2str(ksize) 'clusters the mode classifier made up an average of ' num2str(occurenc) ' % of the total values for each cluster'])
		break
	end

%predict and get accuracy
	predict_test = assign_cluster(mytest, centers);
	assign_test = predict_test;
	m=length(predict_test);
	for i=1:m
		assign_test(i) = map(assign_test(i));
	end
	accuracy = sum(predict_test == mytest(:,end))/m;
	disp(['TEST SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(accuracy) ' in the test set'])

	predict_cv = assign_cluster(cv, centers);
	assign_cv = predict_cv;
	m=length(predict_cv);
	for i=1:m
		assign_cv(i) = map(assign_cv(i));
	end
	accuracy = sum(predict_cv == cv(:,end))/m;
	disp(['CV SET RESULTS: With ' num2str(ksize) ' clusters we had an accuracy of ' num2str(accuracy) ' in the test set'])

	csvwrite([num2str(ksize)  '.centers.kmeans.csv' ], [ map' centers ])
	csvwrite([num2str(ksize) '.train.results.kmeans.csv'],[ mytrain(:,1) predict_train ])
	csvwrite([num2str(ksize) '.test.results.kmeans.csv'],[ mytest(:,1) predict_test ])
	csvwrite([num2str(ksize) '.cv.results.kmeans.csv'],[ cv(:,1) predict_cv ])
	ksize = ksize + 1;
end
