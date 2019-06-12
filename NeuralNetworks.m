%% CA6-Machine Learning Part 2: Neural Networks

% Title : Neural Networks
%
% SUMMARY: 
%
% INPUT: Matrice :
%
% OUTPUT:  : Matrice all_theta : [class*weights]
%
% Made by: Judicaël Fassaya
% Date: May 12th, 2019


%% Inialisation
clear ; close all; clc % clear current workspace
%% Setup the parameters
lambda = 0.1;
input_layer_size  = 2;  % number of input features
hidden_layer_size = 2;   % hidden units
num_labels = 2;          % number of labels 
                          

% %% =========== Part 1: Loading and Visualizing Data =============
% %  We start the exercise by first loading and visualizing the dataset. 
% %
% Load Training Data
% convert 3D dataset into 2d matrix

[X,y] = extract_data('01cr.fdt');
[X] = data_redux(X);
[rows, column] = size(X);
y = y';
left = find(y==1); right = find(y == 2);
m = size(X, 1);

% Plot Examples

plot(X(left, 1), X(left, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
hold on
plot(X(right, 1), X(right, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

xlabel('mean voltage per trial in electrode L-HEOG')
ylabel('mean voltage per trial in electrode R-HEOG')
title('distribution of ocular electrodes mean voltage as a function of electrode channel')

legend('left saccadic eye movement','right saccadic eye movement')

fprintf('Program paused. Press enter to continue.\n');
pause; % pause the program

%% ================ Part 2: Loading Pameters ================
% In this part, we compute neural network parameters.

% Compute the weights into variables Theta1 and Theta2

Theta1 = oneVsAll(X, y, num_labels, lambda) % ok

% intermediary
a1 = [ones(m, 1) X];
% concatenate to add bias terms
%size(a1)
%size(Theta1)

z2 = a1 * Theta1';
a2 = sigmoid(z2);
%
Theta2 = oneVsAll(a2, y, num_labels, lambda) 
% computed from activation units value of previous layer 

%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. We now implement the "predict" function to use the
%  neural network to predict the labels of the training set. It allows
%  to compute the training set accuracy.

pred = predict(Theta1, Theta2, X);
predictions = pred % vector containing matched classes predicitons

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  Randomly permute examples
rp = randperm(m); 

for i = 1:m
    % Display randomly
    fprintf('\nDisplaying ERP datas matching for direction\n');
    % set a random number in range of examples
    example = randi([1,length(X)])
    
    plot(X(example,1),X(example,2),'-s','MarkerSize',12)
    dataexample = [X(example,1),X(example,2)]
    % displayDatapoint on plot, precise point
    
    if y(example) == 1
        fprintf('\nCorresponding direction : left (1)\n');
    else
         fprintf('\nCorresponding direction : right (2)\n');
    end
      
    
    pred = predict(Theta1, Theta2, X(example,:));
    fprintf('\nNeural Network Prediction: %d', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

correct = 0; %Initialize correct trials

% Accuracy percentage
for trial = 1:Num_trials 
     if prediction(trial) == labels_test(trial)%If prediction matches label, mark correct response
                correct = correct + 1;
     end
end
correct = (correct/Num_trials)*100; %Convert to a percentage 

