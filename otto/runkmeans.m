function [ idx centers map ] = runkmeans(data, clusters, iters)

%Train kmeans centers
[idx, centers] = kmeans(data,clusters, 'MaxIter',iters);
map=zeros(clusters);
for i=1:clusters
        map(i) = mode(data(idx == i, end)); %Create map the most popular classnum of each cluster to the clusternum
end


end
