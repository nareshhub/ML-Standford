function p = predict(theta1, theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(theta2, 1);
X = [ones(m,1) X]
% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

layer1 = X*theta1';
sig1 = 1./(1+exp(-layer1));
sig1=[ones(length(sig1),1) sig1];
layer2 = sig1*theta2';
sig2 = 1./(1+exp(-layer2));

[A,p] = max(sig2, [], 2);







% =========================================================================


end
