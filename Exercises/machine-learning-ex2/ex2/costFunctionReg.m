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

%sigmoid function 
z=X*theta;
h=1./(1+exp(-z));

%costFunction.m copy
y0base=-transpose(y)*log(h);
y0=sum(y0base);
y1base=transpose(1-y)*(log(1-h));
y1= sum(y1base);
partJ0=(y0-y1)/m;

error=transpose(X)*(h-y);
partGrad0=(1/m)*error;

%Addition to csotFuntion.m: unique to costFunctionReg.m
theta(1,1)=0;
sumtheta=sum(theta.^2);
partJ1=(lambda/(2*m))*sumtheta;
J=partJ0+partJ1;

partGrad1=(lambda/m)*(theta);
grad=partGrad0+partGrad1;





% =============================================================

end
