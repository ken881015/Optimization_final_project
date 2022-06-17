function stop = plotUpdate(x, optimValues, state)

% Output function that plots the iterates of the optimization algorithm.

% Copyright 2010-2016, The MathWorks, Inc.

x = x(:);
x = x';

stop = false;

switch state
    case 'init'
        % Plot objective function contours
        PlotContour(x,optimValues.fval);
    case 'iter'
        if ~(optimValues.iteration == 0)
            % Update contour plot to show current solution
            PlotPoint(x,optimValues.fval);
        end
    case 'done'
        if ~(optimValues.iteration == 0)
            % Finish this plot and set up for the next
            PlotEnd(x,optimValues.fval)
        end
end

% -------------------------------------------------------------------------
% helper function PlotContour
% -------------------------------------------------------------------------
function PlotContour(x,z)

% Check to see if figure exists, if not create it
h = findobj('Tag','PlotIterates');
old = false;
if isempty(h)
    h = figure('Tag','PlotIterates');
    
%     fobjfcn = @(x,y) log( 1 + 3 * (y - (x.^3 - x)).^2 + (x - 4/3).^2 );
%     fcontour(fobjfcn, [-2.5 2.5], 'LevelStep', 0.3);

    fobjfcn = @(x,y) x^2+2*y^2-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7;
    fcontour(fobjfcn, [-100 100], 'LevelStep', 0.3);
        
    % Colors to use for multiple plots
    ms.index = 1;
    ms.Colors = ['mgcryk'];
    set(h,'UserData',ms);   
end

PlotPoint(x,z)

% -------------------------------------------------------------------------
% helper function PlotPoint
% -------------------------------------------------------------------------
function PlotPoint(x,z)

% Check to see if figure exists, if not, create
h = findobj('Tag','PlotIterates');
if isempty(h)
    PlotContour(x,z)
    h = gcf;
else
    % Update plot with new point
    figure(h)
    ms = get(h,'UserData');
    hold on
    ipts = findobj('Tag','SurfacePoints');
    if isempty(ipts)
        ipts = plot(x(:,1),x(:,2),'MarkerFaceColor','b',...
            'MarkerSize',10,...
            'Marker','o',...
            'MarkerEdgeColor',ms.Colors(ms.index),...
            'Color','k','LineWidth',2);
        set(ipts,'Tag','SurfacePoints');
    else
        set(ipts, 'XData', [get(ipts,'XData'),x(:,1)']);
        set(ipts, 'YData', [get(ipts,'YData'),x(:,2)']);
    end
end

% -------------------------------------------------------------------------
% helper function PlotEnd
% -------------------------------------------------------------------------
function PlotEnd(x,z)

% Check to see if figure exists, if not, create
h = findobj('Tag','PlotIterates');
if isempty(h)
    PlotContour(x,z)
    h = gcf;
end

% Change to next color
ms = get(h,'UserData');
ms.index = ms.index+1;
set(h,'UserData',ms);

% Tag already plotted points with new tag to preserve them
% 'SurfacePoints' tag will be used for next plot
spts = findobj('Tag','SurfacePoints');
set(spts,'Tag',['SurfacePoints',num2str(ms.index)]);
