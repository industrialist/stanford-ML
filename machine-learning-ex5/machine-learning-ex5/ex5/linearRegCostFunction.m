function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% --- Calculate Cost --- %
thetaReg = theta.^2;
%set bias term to zero so we can use this in the summation
thetaReg(1) = 0;
J = sum((X*theta - y).^2) + lambda*(sum(thetaReg));
J = J/(2*m);

% --- Calculate Gradient --- %

n = size(theta,1);
thetaReg2 = theta;
thetaReg2(1) = 0;

for j = 1:n
    grad(j) = sum((X*theta - y).*X(:,j)) + lambda*thetaReg2(j);
end

grad = grad/m;


% =========================================================================

grad = grad(:);

end
