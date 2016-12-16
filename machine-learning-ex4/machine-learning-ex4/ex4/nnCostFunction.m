function [J, grad] = nnCostFunction(nn_params, ...
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




%Feedforward:
% add a column of 1's to the input layer, X
X = [ones(m,1) X];

% calculate the activations of layer 2
A2 = sigmoid(X*Theta1');
A2 = [ones(m, 1) A2]; 

%test = A2*Theta2';
%test(1,:)

% calculate the activations of layer 3
H  = sigmoid( A2*Theta2' );

% process the y-vector into a matrix so that we can traverse over
% all training examples to sum up the cost
y_matrix = eye(num_labels);
y_matrix = y_matrix(y,:);
y_matrix = y_matrix';




% for i = 1 to m, calculate the cost (scalar)
for i = 1:m
    %fprintf('H row; 1-H row; and log(1-Hrow)');
    %H(i,:)
    %1 - H(i,:)
    %log(1 - H(i,:))
    J = J - log(H(i,:))*y_matrix(:,i) - (log(1 - H(i,:))*(1-y_matrix(:,i)) );
end

J = J/m;






% add regularisation term:
reg1 = 0;
reg2 = 0;

% element-wise square of each term in the parameter matrices
Theta1Temp = Theta1.^2;
Theta2Temp = Theta2.^2;

% for every row,
for i = 1:size(Theta1,1)
    % sum every element in that row
    reg1 = reg1 + sum(Theta1Temp(i,2:end));
end

% for every row,
for i = 1:size(Theta2,1)
    % sum every element in that row
    reg2 = reg2 + sum(Theta2Temp(i,2:end));
end

J = J + lambda*(reg1 + reg2)/(2*m);





D2 = 0;
D3 = 0;
%y_matrix = y_matrix';

% Backpropagation Algorithm - used to determine Theta1_grad and Theta2_grad
% Note that X already has had the bias units added.

%classVector = [10; 1; 2; 3; 4; 5; 6; 7; 8; 9];

%classVector = zeros(num_labels,1);
%classVector(num_labels) = num_labels;
%for i = 1:num_labels
%    classVector(i) = i;
%end

%fprintf('size of X, m');
%size(X)
%size(m)

for i = 1:m
    % step 1: compute feed-forward for a single training example on all
    % nodes in all layers: note, size(X(i,:)) = 1x401 .. (row vector)
    
    z2 = X(i,:)*Theta1';
        % Theta1 ~ 25x401
        % z2 ~ 1x25. Activate and add bias unit to activation vector:
    a2 = [1 sigmoid(z2)];
    z3 = a2*Theta2';
        % Theta2 ~ 10x26
        % z3 ~ 1x10
    a3 = sigmoid( z3 );
    
    % step 2: for each output unit k in layer 3, set deltak = ak - yk....
    % In other words, calculate the error in the output vector a3 to be
    % equal to a3, but subtract a 1 from the desired output value...
    % Our given y value is a scalar - need to conver to a vector with a
    % "logical array":
    
    %size(a3')
    %size( y(i) == classVector )
    %d3 = a3' - ( y(i) == classVector );
    % recall the outputs of the sigmoid are:   0 > a3 > 1
    d3 = a3' - ( y_matrix(:,i) );
    
    % step 3: compute the error term for layer 2:
    % the sigmoid below produces a row vector; d2 is column vector so we
    % need to transpose g for the element-wise multiplication, and add 1 to
    % g otherwise the matrix dimensions wont agree (see outputs of size()
    % below:
    
    g = sigmoidGradient(z2);
        %size(Theta2')
        %size(d3)
        %size(g')
    d2 = (Theta2(:,2:end))'*d3.*g';
    
    % step 4: now that we have all the error terms in all the layers for
    % this training example, can accumulate the errors together before
    % moving onto the next training example. Note that these terms are
    % matrices.
 
        %fprintf('size d3 = ');
        %size(d3)       %
        %fprintf('size a2 = ');
        %size(a2)       %
        %fprintf('size d2 = ');
        %size(d2)       %
        %fprintf('size X(i,:) = ');
        %size(X(i,:))  %

    D3 = D3 + d3*a2;
    D2 = D2 + d2*X(i,:);
    
end

%fprintf('D2 is (before regularisation)');
%D2

%fprintf('D3 is (before regularisation)');
%D3

%fprintf('size of D3');
%size(D3)
%fprintf('size of Theta2');
%size(Theta2)
%fprintf('size of D2');
%size(D2)
%fprintf('size of Theta1');
%size(Theta1)

Theta1Temp = (lambda/m)*Theta1(:,2:end);
Theta2Temp = (lambda/m)*Theta2(:,2:end);

Theta1_grad = D2/m + [zeros(size(Theta1(:,1),1),1) Theta1Temp];
Theta2_grad = D3/m + [zeros(size(Theta2(:,1),1),1) Theta2Temp];

%fprintf('size of D2:');
%size(D2)
%fprintf('size of D3:');
%size(D3)



%fprintf('D2 is (after regularisation)');
%D2

%fprintf('D3 is (after regularisation)');
%D3



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
