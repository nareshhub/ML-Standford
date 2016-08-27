function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
thsize = length(theta);
hyp = X*theta;
hyp1 = 1./(1+exp(-hyp));
diff = (log(hyp1).*(-y))-((log(1-hyp1).*(1-y)));
theta1 = theta(2:thsize);
thetasqr = sum(theta1.^2);
J = (1/m)*sum(diff)+((lambda/(2*m))*thetasqr);

diff1 = (hyp1-y);
diff2=diff1';

for j=1:thsize
if j == 1
    cost1=diff2*X(:,j);
    grad(j) = (1/m)*sum(cost1);
else
    cost2=diff2*X(:,j);
    grad(j) = (1/m)*sum(cost2)+((lambda/m)*theta(j));
end
end
% =============================================================

end
