%Take all the learned paramaters and test data
%iterate row by row of the test data and predict values
%Clear out any existing solution files and write headers/data
function makesolution2(test, theta, all_theta, centers, classifiers, map, only_logistic, kmeans, logistic)
[rows n] = size(test);
all_solution = zeros(rows,classifiers+1);

%TODO iterate through solution line at a time
%erase file
header = 'id';

for i=1:classifiers
	header = [header ',' 'Class_' num2str(i)];
end
f1 = fopen('solution.results.logistic.csv', 'w');
f2 = fopen([num2str(length(centers)) '.solution.results.kmeans.csv'], 'w');
f3 = fopen([num2str(length(centers)) '.solution.results.kmeans.and.logisitic.csv'], 'w');
fprintf(f1, [header '\n']);
fprintf(f2, [header '\n']);
fprintf(f3, [header '\n']);
fclose(f1);
fclose(f2);
fclose(f3);

for row=1:rows
m=1;

if only_logistic
	solution = log_predict_sol(theta,[1 test(row,2:end)], classifiers);
	dlmwrite('solution.results.logistic.csv', [test(row,1) solution ],'-append')
end
if  kmeans
	[ accuracy predict mapped ] = assess_kmeans(test(row,2:end), -1, centers, map);
	dlmwrite([num2str(length(centers)) '.solution.results.kmeans.csv'],[ test(row,1) (mapped == 1:classifiers) ], '-append')
end
if (logistic && kmeans)
	for j=1:size(centers,1) %Run a sepperate regression against each cluster
		if predict == j
			this_theta = ((j-1)*classifiers+1):(j*classifiers);
			solution = log_predict_sol( all_theta(:,this_theta),[1 test(row,2:end)], classifiers);
			break
		end
	end
	dlmwrite([num2str(length(centers)) '.solution.results.kmeans.and.logisitic.csv'],[ test(row,1) solution], '-append')
end

end


