function [ idx centers map ] = runkmeans(data, y, clusters, iters)

%Train kmeans centers
[idx, centers] = kmeans(data,clusters, 'MaxIter',iters);
map=zeros(clusters);

for i=1:clusters
        map(i) = mode(y(idx == i)); %Create map the most popular classnum of each cluster to the clusternum
end
end
