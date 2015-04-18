function [idx] = assign_cluster(data, centers)
size(data)
size(centers)

%assign test data to clusters
[m n] = size(data);
[K L] = size(centers);
distance = zeros(m,K);
for i=1:K
    for j=1:L
       	distance(:,i) = distance(:,i) + (data(:,j) - centers(i,j)) .^ 2;
    end
end

%TODO: this seems inefficient
[tmp idx] = min(distance');
idx = idx';

end

