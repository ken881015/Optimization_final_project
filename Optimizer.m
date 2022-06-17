clear;
clc;

%% Objective Function
% fcn = @(x, y) log(1+3*(y - (x.^3 - x)).^2 + (x - 4/3).^2 );
% fcn = @(x,y) x^2+2*y^2-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7; % Quetion1
fcn = 


%% Plot the Objective
fsurf(fcn, [-100 100], 'ShowContours', 'on')
view(127, 38)

%% Create optimization problem
prob = optimproblem;

%% Define Varaiables
x = optimvar('x', 'LowerBound', -2.5, 'UpperBound', 2.5);
y = optimvar('y', 'LowerBound', -2.5, 'UpperBound', 2.5);

%% Define objective function
prob.Objective = log(1+3*(y - (x.^3 - x)).^2 + (x - 4/3).^2 );

%% Set optimization options, and optimize
initialpt.x = 2;
initialpt.y = 2;

iteration = 21;

options = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter', 'OutputFcn', @plotUpdate);
% options = optimoptions('fmincon','Algorithm','interior-point','Display','iter', 'OutputFcn', @plotUpdate);
% options = optimoptions('fmincon','Algorithm','sqp','Display','iter', 'OutputFcn', @plotUpdate);
options = optimoptions('linprog','Algorithm','dual-simplex','Display','iter');


%% Solve Prob
[sol, fval, exitflag, output] = solve(prob, initialpt, 'Options', options);

%% Disp
disp("x = "+ sol.x+ ", "+ "y = "+ sol.y);











