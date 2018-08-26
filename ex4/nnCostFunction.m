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
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_modified = zeros(m, num_labels);
temp_grad1 = 0;
temp_grad2 = 0;

for i = 1:m,
   a1 = [1, X(i, :)]';
   z2 = Theta1 * a1;
   a2 = sigmoid(z2); 
   a2 = [1; a2];
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);
   y_modified = zeros(num_labels, 1);
   y_modified(y(i)) = 1;

   delta3 = a3 - y_modified;
   delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2);

   temp_grad1 = temp_grad1 + delta2 * a1';
   temp_grad2 = temp_grad2 + delta3 * a2';

   J = J + sum((y_modified .* log(a3)) + (1 - y_modified) .* log(1 - a3));
end;

J = (-1/m)*J;
Theta1_grad = (1/m) * temp_grad1;
Theta2_grad = (1/m) * temp_grad2;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:, 2:end);

fsum = 0;

for j = 1:hidden_layer_size,
   for k = 2:input_layer_size+1,
	fsum = fsum + (Theta1(j, k))^2;
   end;
end;

for j = 1:num_labels,
   for k = 2:hidden_layer_size+1,
	fsum = fsum + (Theta2(j, k))^2;
   end;
end;
J = J + (lambda/(2*m))*fsum;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
