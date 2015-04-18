function [ accuracy predict mapped ] = assess_kmeans(data, y, centers, map)
	[m n] = size(data);
        predict = assign_cluster(data, centers);
        mapped = predict;
        m=length(predict);
        for i=1:m
                mapped(i) = map(mapped(i));
        end
        accuracy = sum(predict == y)/m;
end
