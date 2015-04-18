function [ accuracy predict mapped ] = assess_kmeans(data, y, centers, map)
        predict = assign_cluster(data, centers);
        mapped = map(predict);
        accuracy = sum(mapped == y)/length(predict);
end
