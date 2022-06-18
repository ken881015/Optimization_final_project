clear;
clc;

syms x y;

%% Select the Objective Function and their bound to solve
problem = input('Input your problem(1~7 default:1):');
if isempty(problem)
    problem = 1;
end

switch problem
    case 1
        fcn = @(x,y) x.^2+2*y.^2-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7; % Quetion1
        fcn_sym = symfun(x.^2+2*y.^2-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7,[x,y]);
        bnd = [-100,100,-100,100];
    case 2
        fcn = @(x,y) 2.*x.^2 - 1.05.*x.^4 + x.^6./6. + x.*y + y.^2;
        fcn_sym = symfun(2.*x.^2 - 1.05.*x.^4 + x.^6./6. + x.*y + y.^2,[x,y]);
        bnd = [-5,5,-5,5];
    case 3
        fcn = @(x,y) (x-1).^2 + 35*(2.*y.^2-x).^2;
        fcn_sym = symfun((x-1).^2 + 35*(2.*y.^2-x).^2,[x,y]);
        bnd = [-10,10,-10,10];
    case 4
        fcn = @(x,y) (4-2.1.*x.^2+(x.^4)/3.).*x.^2+x.*y+(-4+4.*y.^2).*y.^2;
        fcn_sym = symfun((4-2.1.*x.^2+(x.^4)/3.).*x.^2+x.*y+(-4+4.*y.^2).*y.^2,[x,y]);
        bnd = [-3,3,-2,2];
    case 5
        fcn = @(x,y) (100*sqrt(sqrt((y-0.01*x.^2).^2)) + 0.01*sqrt((x+10).^2));
        fcn_sym = symfun(100*(abs(y-0.01*x.^2)).^(0.5)+0.01*abs(x+10),[x,y]);
        bnd = [-15,5,-3,3];
    case 6
        fcn =  @(x,y) (1+(x+y+1).^2.*(19-14*x+3*x.^2-14*y+6*x.*y+3*y.^2)) .* (30+(2*x-3*y).^2.*(18-32*x+12*x.^2+48*y-36*x.*y+27*y.^2));
        fcn_sym = symfun((1+(x+y+1).^2.*(19-14*x+3*x.^2-14*y+6*x.*y+3*y.^2)) .* (30+(2*x-3*y).^2.*(18-32*x+12*x.^2+48*y-36*x.*y+27*y.^2)),[x,y]);
        bnd = [-2,2,-2,2];
    case 7
        fcn = @(x,y) -(1+cos(12*(x.^2+y.^2).^(0.5))) ./ (0.5*(x.^2+y.^2) + 2) ;
        fcn_sym = symfun(-(1+cos(12*(x.^2+y.^2).^(0.5))) / (0.5*(x.^2+y.^2) + 2),[x,y]);
        bnd = [-5.12,5.12,-5.12,5.12];
    otherwise
end

%% Plot the Objective - requirement 1
fsurf(fcn, bnd, 'ShowContours', 'on')
view(127, 38)

%% Create optimization problem
prob = optimproblem;

% Define Varaiables
x = optimvar('x', 'LowerBound', bnd(1), 'UpperBound', bnd(2));
y = optimvar('y', 'LowerBound', bnd(3), 'UpperBound', bnd(4));

% Define objective function
switch problem
    case 1
        prob.Objective = x.^2+2*y.^2-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7;
    case 2
         prob.Objective = 2*x.^2 - 1.05*x.^4 + x.^6/6. + x*y + y.^2;
    case 3
         prob.Objective = (x-1).^2 + sum((2:8)*(2*y.^2-x).^2);
    case 4
         prob.Objective = (4-2.1*x.^2+(x.^4)/3.)*x.^2+x*y+(-4+4*y.^2)*y.^2;
    case 5
         prob.Objective = (100*sqrt(sqrt((y-0.01*x.^2).^2)) + 0.01*sqrt((x+10).^2));
    case 6
         prob.Objective = (1+(x+y+1).^2.*(19-14*x+3*x.^2-14*y+6*x.*y+3*y.^2)) .* (30+(2*x-3*y).^2.*(18-32*x+12*x.^2+48*y-36*x.*y+27*y.^2));
    case 7
         prob.Objective = -(1+cos(12*(x.^2+y.^2).^(0.5))) / (0.5*(x.^2+y.^2) + 2) ;
    otherwise
end
%% Set optimization options, and optimize
iteration = 60;

Algorithms=[
%     optimoptions('fminunc','Algorithm','quasi-newton','Display','iter', 'OutputFcn', @plotUpdate)
%     optimoptions('fmincon','Algorithm','interior-point','Display','iter', 'OutputFcn', @plotUpdate1)
%     optimoptions('fmincon','Algorithm','sqp','Display','iter', 'OutputFcn', @plotUpdate)
%     optimoptions('linprog','Algorithm','dual-simplex','Display','iter')
];


%% Solve Prob
% for initpt = 1:1
%     initialpt.x = randi(bnd(2)-bnd(1),1,1)+bnd(1);
%     initialpt.y = randi(bnd(4)-bnd(3),1,1)+bnd(3);
% 
%     for alg = 1:length(Algorithms)
%         disp(Algorithms(alg).Algorithm)
%         [sol, fval, exitflag, output] = solve(prob, initialpt, 'Options', Algorithms(alg));
%         
%         disp("initial place: x = "+initialpt.x +  ", y = "+ initialpt.y)
%         disp("optimized    : x = "+ sol.x+ ", "+ "y = "+ sol.y);
%         
%     end
% end


%%  Use Symbolic Function
[f_gradient,f_hessian,f_approx,f_gradient_cg] = funcs2(fcn_sym);


%% Function Declaration -  Calculate the information for the Optimization Method
function [f_gradient,f_hessian,f_approx,f_gradient_cg] = funcs2(f)
    syms x1 x2 a b c
    f_gradient = gradient(f); %gradient_descent, newton, quasi_bfgs
    f_hessian = hessian(f, [x1,x2]); %newton method
    f_approx = taylor(f, [x1,x2],[a b],'Order', 2); % taylor expansion of f for quadratic behavior
    f_approx = symfun(f_approx, [x1 x2 a b]);
    f_gradient_cg = gradient(f_approx, [x1 x2]); % nonlinear CG parameter
end













