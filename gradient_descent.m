%% Gradient descent
function [point, output, time, iter, loss, points_array] = gradient_descent(f, f_gradient, point, epsilon, max_iter,up,low)

tic;
loss = [];
points_array = [];

p = num2cell(point);
before = f(p{:}); % initial function value
step_size= 0.01;
for iter=1:max_iter
    if (iter == max_iter)
        % when reach to end of loop, evoke error
        %error('Steepest gradinet descent failed!');
    end

    % compute gradient
    p = num2cell(point);
    gradient = double(f_gradient(p{:}));
    xlast  = point;
    glast = gradient;
    
    % inexact line search for the given search direction (-gradient)
    % step_size = 5 * rand(1); % step_size initialization
    step_size = inexact_search(f, point, 0.001, gradient, -gradient, max_iter,up,low);
    
    a = transpose(step_size*gradient);
    
    % update point
    point = point - a;
    
    % calculate delta x and delta gradient
    p = num2cell(point);
    delta_x = transpose(point - xlast);
    delta_g = double(f_gradient(p{:})) - glast;
    step_size2 = (transpose(delta_g)*delta_x)/(transpose(delta_g)*delta_g);

    % clip the value that beyond bound
    point = clip(point,up,low);
    
    % terminate condition
    p = num2cell(point);
    
    % Point Store
    points_array = cat(1, points_array, point);
    
    current = f(p{:});
    
    % loss
    loss(iter) = eval(abs(current - before));
    
    if(abs(current - before) <= epsilon)
        break;
    end;
    before = current;
end;

output = double(current);
time = toc;

