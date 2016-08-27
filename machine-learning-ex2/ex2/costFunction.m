function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hyp = X*theta;
hyp1 = 1./(1+exp(-hyp));
diff = (log(hyp1).*(-y))-((log(1-hyp1).*(1-y)));
J = (1/m)*sum(diff);

diff1 = (hyp1-y);
diff2=diff1';
cost1=diff2*X(:,1);
cost2=diff2*X(:,2);
cost3=diff2*X(:,3);
grad(1) = (1/m)*sum(cost1);
grad(2) = (1/m)*sum(cost2);
grad(3) = (1/m)*sum(cost3);



% =============================================================

end
