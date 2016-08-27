function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

[m n]=size(X);
[m1 n1]=size(Theta);
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

pred = X*Theta';
pred1 = pred;
out = ((pred1-Y).*R);
J = sum(sum(out.^2))/2;
J = J + lambda/2*(sum(sum(Theta.^2))+sum(sum(X.^2)));
out1 = out*Theta;
X_grad = out1+(lambda*X);
Theta_grad = out'*X+(lambda*Theta);


% for i =1:n
%     idx = find(R(i,:)==1);
%     Theta1 = Theta(idx,:);
%     Y1 = Y(i,idx);
%     X_grad(i,:) = (X(i,:)*Theta1'-Y1)*Theta1;
% end

% for i =1:n1
%     idx = find(R(i,:)==1);
%     Theta1 = Theta(idx,:);
%     X1 = X(idx,i);
%     Theta_grad(i,:) = (X(i,:)*Theta1'-Y1)*X1;
% end










% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
