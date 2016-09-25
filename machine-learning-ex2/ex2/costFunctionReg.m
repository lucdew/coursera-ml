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




%myhyp = sigmoid(X*theta);

% J = (1/m) * ( -y'*log(myhyp) - ((1-y)'* log( 1-myhyp)));

% [myCost,myGrad]=costFunction(theta,X,y);

myhyp = sigmoid(X*theta);
% myCost = (1/m) * ( -y'*log(myhyp) - ((1-y)'* log( 1-myhyp)));
 myCost = (1/m) * sum( -y.*log(myhyp) - ((1-y).* log( 1-myhyp)));
myGrad = (1/m) * (X'*(myhyp -y ));


myreg = ones(size(theta),1);
myreg(1)=0;

myfac = lambda / ( 2*m);

sqtheta=theta'*theta;


J = myCost + (myfac * sqtheta);
J(1)=myCost(1);




grad = myGrad + ( lambda / m * theta);
grad(1)=myGrad(1);







% =============================================================

end
