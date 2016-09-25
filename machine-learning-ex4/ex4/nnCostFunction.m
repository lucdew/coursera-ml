function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];



for i=1:m
   A1=X(i,:)';
   yk=y(i,:)';
   
   A2=sigmoid(Theta1*A1);  
   A2 = [1;A2];
   yvec=1:num_labels;
   yvec=(yvec==yk)';
   h=sigmoid(Theta2*A2);   
   J=J+sum(-yvec.*log(h)-((1-yvec).*log(1-h)));
end

J=(1/m)*J;

% with regularization
Theta1R=Theta1;
Theta1R(:,1)=0;
Theta2R=Theta2;
Theta2R(:,1)=0;
J=J+lambda*(sum(sum(Theta1R.*Theta1R))+sum(sum(Theta2R.*Theta2R)))/(2*m);



% back propagation step1

for i=1:m
   A1=X(i,:)';
   yk=y(i,:)';
   yvec=1:num_labels;
   yvec=(yvec==yk)';  
   
   Z2=Theta1*A1; 
   A2=sigmoid(Z2);  
   A2 = [1;A2];
   
   Z3=Theta2*A2;
   A3=sigmoid(Z3);
   
   delta_3=(A3-yvec);
   
   TD=Theta2'*delta_3;
   
   delta_2=TD.*sigmoidGradient([1;Z2]);
   
   delta_2=delta_2(2:end);
   Theta1_grad = Theta1_grad + delta_2*A1';
   Theta2_grad = Theta2_grad + delta_3*A2';
   

end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;


Theta1_grad = Theta1_grad + lambda/m*Theta1R;
Theta2_grad = Theta2_grad + lambda/m*Theta2R;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
