%ml-008 Exercise2
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/15/2015
function g = sigmoid(z)

% You need to return the following variables correctly 
g = zeros(size(z));
[m, n] = size(z); %Z can be a scalar, vecotr, or matrix

%Return the
for i=1:m
    for j=1:n
        g(i,j) = 1/(1+(exp(1)^(-z(i,j))));
    end
end
