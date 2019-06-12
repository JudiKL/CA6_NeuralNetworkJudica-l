%% CA6-Machine Learning : One-vs-all
%
% Title : Training OneVsAll
%
% SUMMARY: Implement Logistic Regression to One-vs-all classification.
% Execute a prototypic training for given dataset
%
% INPUT: Matrice : [all examples*all features]
%
% OUTPUT:  : Matrice all_theta : [class*weights]
%
% Made by: Judicaël Fassaya
% Date: May 12th, 2019


%% Initialization

clear ; close all; clc % clear current workspace 

%% Setup of the parameters

input_layer_size  = 2;  % number of initial features
num_y = 2;          % number of desired classes  

%% =========== Part 1: Loading and Visualizing Data =============
%  We start by loading and visualizing the dataset.

% Load Training Data
% convert 3D dataset into 2d matrix

[X,y] = extract_data('01cr.fdt');
[X] = data_redux(X);
[rows, column] = size(X);
y = y';
left = find(y==1); right = find(y == 2);

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

%% ============ Part 2: One-vs-All Training ============
% Compute weights enabling feedforward from first to second layer
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1; % regularization term (optionnal : in case of increased number of features)
[all_theta] = oneVsAll(X, y, num_y, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ================
% predict classes based on the first inputs
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

