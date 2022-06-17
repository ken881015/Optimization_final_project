%% 
clc;
clear;
close all;
max_iter = 500; % when reach to end of loop, evoke error      
epsilon = 0.00001;  % for terminating condition

%% Problem 
% - find minima f(x1, x2, x3) = x1^2+x2^2+x3^2+1000*(exp(-(x1^2+x2^2+x3^2)))
% - find minima f(x1, x2, x3) = 54*x1^2+44*x2^2+72*x3^2+2*x1*x2-88*x1*x3-42*x2*x3+76*x1-26*x2-130*x3+123

syms x1 x2 x3 a b c;

pointa = [-1.516646, 1.516646, -1.516646]; % initial point for fa
pointb = [0.5,1,1.5];% initial point for fb

fa = symfun(x1^2+x2^2+x3^2+1000*(exp(-(x1^2+x2^2+x3^2))), [x1 x2 x3]); 
fa2 = @(x) x(1)^2+x(2)^2+x(3)^2+1000*(exp(-(x(1)^2+x(2)^2+x(3)^2))); 

fb = symfun(54*x1^2+44*x2^2+72*x3^2+2*x1*x2-88*x1*x3-42*x2*x3+76*x1-26*x2-130*x3+123 , [x1 x2 x3]); 
fb2 = @(x) 54*x(1)^2+44*x(2)^2+72*x(3)^2+2*x(1)*x(2)-88*x(1)*x(3)-42*x(2)*x(3)+76*x(1)-26*x(2)-130*x(3)+123; 

ulimit = [inf,inf,inf];
llimit = [-inf,-inf,-inf];

ua = [100 , 10 , 100];
la = -ua;

ub = ua;
lb = la;
option=optimset('Display','off');

fprintf('=============== Question 1 ================\n');
[f_gradient,f_hessian,f_approx,f_gradient_cg] = funcs3(fa);
Output(fa,f_gradient,f_hessian,f_approx,f_gradient_cg,pointa,epsilon, max_iter,ua,la);

options = optimset('Display','iter');

tic;
fminsearch(fa2,pointa,options);
toc;

tic;
fminunc(fa2,pointa,options);
toc;

fprintf('=============== Question 2 ================\n');
[f_gradient,f_hessian,f_approx,f_gradient_cg] = funcs3(fb);
Output(fb,f_gradient,f_hessian,f_approx,f_gradient_cg,pointb,epsilon, max_iter,ub,lb);
tic;
fminsearch(fb2,pointb,options);
toc;

tic;
fminunc(fb2,pointb,options);
toc;

function [f_gradient,f_hessian,f_approx,f_gradient_cg] = funcs3(f)
    syms x1 x2 x3 a b c
    f_gradient = gradient(f); %gradient_descent, newton, quasi_bfgs
    f_hessian = hessian(f, [x1,x2,x3]); %newton method
    f_approx = taylor(f, [x1,x2,x3 ],[a b c ],'Order', 2); % taylor expansion of f for quadratic behavior
    f_approx = symfun(f_approx, [x1 x2 x3 a b c]);
    f_gradient_cg = gradient(f_approx, [x1 x2 x3]); % nonlinear CG parameter
end

function Output(f, f_gradient, f_hessian,f_approx,f_gradient_cg, point, epsilon, max_iter,up,low)
    [min_point, output, time, iter, loss, points_array] = gradient_descent(f, f_gradient, point, epsilon, max_iter,up,low);
    mintemp = num2cell(min_point);
    fprintf('1. Steepest Gradient Descent - It takes %fsec to generate minimum [%f] in (x1,x2,x3) = (%f,%f,%f) for %d iter\n', time, output, mintemp{:}, iter);
    
    figure();
    x_ticks = linspace(0, length(loss), length(loss));
    plot(x_ticks, loss, 'LineWidth', 3);
    xlabel('Iteration', 'fontsize', 12,'fontweight','bold');
    ylabel('diff', 'fontsize', 12,'fontweight','bold');
    title('Steepest Gradient Descent_ 梯度變化', 'fontsize', 20,'fontweight','bold');
    
    [min_point, output, time, iter, loss_Newton, points_array_Newton] = newton_method(f, f_gradient, f_hessian, point, epsilon, max_iter,up,low);
    fprintf('2. Newton''s Method - It takes %fsec to generate minimum [%f] in (x1,x2,x3) = (%f,%f,%f) for %d iter\n', time, output, min_point(1), min_point(2),min_point(3), iter);
    
    figure();
    x_ticks = linspace(0, length(loss_Newton), length(loss_Newton));
    plot(x_ticks, loss_Newton, 'LineWidth', 3);
    xlabel('Iteration', 'fontsize', 12,'fontweight','bold');
    ylabel('diff', 'fontsize', 12,'fontweight','bold');
    title('Newton''s Method_ 梯度變化', 'fontsize', 20,'fontweight','bold');    
    
    [min_point, output, time, iter, loss_CG, points_array_CG] = nonlinear_CG(f_approx, f, f_gradient_cg, point, epsilon, max_iter,up,low);
    fprintf('3. Nonlinear CG - It takes %fsec to generate minimum [%f] in (x1,x2,x3) = (%f,%f,%f) for %d iter\n', time, output, min_point(1), min_point(2),min_point(3), iter);
    
    figure();
    x_ticks = linspace(0, length(loss_CG), length(loss_CG));
    plot(x_ticks, loss_CG, 'LineWidth', 3);
    xlabel('Iteration', 'fontsize', 12,'fontweight','bold');
    ylabel('diff', 'fontsize', 12,'fontweight','bold');
    title('Nonlinear CG_ 梯度變化', 'fontsize', 20,'fontweight','bold');     
    
    [min_point, output, time, iter, loss_BFGS, points_array_BFGS] = quasi_newton_bfgs_method(f, f_gradient, point, epsilon, max_iter,up,low);
    fprintf('4. Quasi Newton''s BFGS Method - It takes %fsec to generate minimum [%f] in (x1,x2,x3,) =  (%f,%f,%f) for %d iter\n', time, output, min_point(1), min_point(2),min_point(3), iter);
    
    figure();
    x_ticks = linspace(0, length(loss_BFGS), length(loss_BFGS));
    plot(x_ticks, loss_BFGS, 'LineWidth', 3);
    xlabel('Iteration', 'fontsize', 12,'fontweight','bold');
    ylabel('diff', 'fontsize', 12,'fontweight','bold');
    title('Quasi Newton_ 梯度變化', 'fontsize', 20,'fontweight','bold');    

end