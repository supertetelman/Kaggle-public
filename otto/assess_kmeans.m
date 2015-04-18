function [ accuracy predict mapped ] = assess_kmeans(data, y, centers, map)
	[m n] = size(data);
        predict = assign_cluster(data, centers);
        mapped = predict;
        m=length(predict);
        mapped = map(predict);
        accuracy = sum(mapped == y)/m;
end
