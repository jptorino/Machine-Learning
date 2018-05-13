function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%



% J = (1/m)*sum(-y .* log(h) - (1 - y) .* log(1-h));
new_theta = theta(2:size(theta));
theta_temp = [0;new_theta];


h = sigmoid(X * theta);		
costPos = -y' * log(h);
costNeg = (1 - y)' * log(1 - h);
%J = (1/m) * (costPos - costNeg)  + (lambda / 2*m) *theta_temp' * theta_temp;

J = sum(costPos - costNeg) / m + (sum(theta(2:size(theta)) .^ 2)) * lambda / (2 * m);

% grad_zero = (1/m)*X(:,1)'*(h-y);
% grad_rest = (1/m)*(shift_x'*(h - y)+lambda*new_theta);
% grad      = cat(1, grad_zero, grad_rest);


grad(1) = sum((h - y)' *  X(:,1)) / m;

for theta_index = 2: size(X,2);
    grad(theta_index) = sum((h - y)' *  X(:,theta_index)) / m + lambda / m * theta(theta_index);
end









% =============================================================

grad = grad(:);

end
