clear;
clc;

%% Objective Function and their bound
% fcn = @(x,y) x^2+2*y^2-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7; % Quetion1
% bnd = [-100,100];

% fcn = @(x,y) 2*x^2 - 1.05*x^4 + x^6/6. + x*y + y^2;
% bnd = [-5,5];

% fcn = @(x,y) (x-1)^2 + sum((2:8)*(2*y^2-x)^2);
% bnd = [-10,10];

% fcn = @(x,y) (4-2.1*x^2+(x^4)/3.)*x^2+x*y+(-4+4*y^2)*y^2;
% bnd = [-3,3,-2,2];

% fcn = @(x,y) 100*(abs(y-0.01*x^2))^(0.5)+0.01*abs(x+10);
% bnd = [-15,5,-3,3];

% fcn = @(x,y) (1+(x+y+1)^2*(19-14*x+3*y^2-14*y+6*x*y+3*y^2))*(30+(2*x-3*y)^2*(18-32*x+12*y^2+48*y-36*x*y+27*x^2));
% bnd = [-2,2];

fcn = @(x,y) -(1+cos(12*(x^2+y^2)^(0.5))) / (0.5*(x^2+y^2) + 2) ;
bnd = [-5.12,5.12];


%% Plot the Objective
fsurf(fcn, bnd, 'ShowContours', 'on')
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











