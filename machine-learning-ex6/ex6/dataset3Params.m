function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
trialValues = [0.01 0.03 0.1 0.3 1 3 10 30];


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


% create a 64 x 3 matrix with the results of the predictions on the
% validation set. column1 = C value; column2 = sigma value; column3 = the
% mean of correct predictions (true positives and true negatives)

results = zeros(64,3);
k = 1
for i = 1:size(trialValues, 2)
    i
    for j = 1:size(trialValues, 2)
        j
        % train a new SVM model
        % "x1 and x2 are dummy parameters for the kernel function"
        % model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        model= svmTrain(X, y, trialValues(i),...
            @(x1, x2) gaussianKernel(x1, x2, trialValues(j)));
        
        
        % svmPredict returns a vector of predictions using the model it has been
        % passed.
        predictions  = svmPredict(model, Xval);
        results(k,1) = trialValues(i);
        results(k,2) = trialValues(j);
        
        % note this (predictions ~= yval) returns errors, hence the
        % requirement to use 'min' below to find the lowest error rate, not
        % 'max' for the highest success rate.
        results(k,3) = mean(double(predictions ~= yval));
        k=k+1
    end
end

results
[~,I] = min(results);

results(I(3),:)
C = results(I(3),1)
sigma = results(I(3),2)

% =========================================================================

end
