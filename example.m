%% clear memory, screen, and close all figures
clear all; clc;
%% Process equation x[k] = sys(k, x[k-1], u[k]);
nx = 1;  % number of states
sys = @(k, xkm1, uk) xkm1/2 + 25*xkm1/(1+xkm1^2) + 8*cos(1.2*k)+uk; 
%sys = @(k, xkm1, uk) xkm1/2 + 25*xkm1/(1+xkm1^2) + 8*cos(1.2*k) + uk; % (returns column vector)
%% Observation equation y[k] = obs(k, x[k], v[k]);
ny = 1;                                           % number of observations
obs = @(k, xk, vk) xk^2/20 + vk;                  % (returns column vector)
%% PDF of process noise and noise generator function
nu = 1;                                           % size of the vector of process noise
sigma_u = sqrt(10);
p_sys_noise   = @(u) normpdf(u, 0, sigma_u);
gen_sys_noise = @(u) normrnd(0, sigma_u);         % sample from p_sys_noise (returns column vector)
%% PDF of observation noise and noise generator function
nv = 1;                                           % size of the vector of observation noise
sigma_v = sqrt(1);
p_obs_noise   = @(v) normpdf(v, 0, sigma_v);
gen_obs_noise = @(v) normrnd(0, sigma_v);         % sample from p_obs_noise (returns column vector)
%% Initial PDF
% p_x0 = @(x) normpdf(x, 0,sqrt(10));             % initial pdf
gen_x0 = @(x) normrnd(0, sqrt(10));               % sample from p_x0 (returns column vector)
%% Transition prior PDF p(x[k] | x[k-1])
% (under the suposition of additive process noise)
% p_xk_given_xkm1 = @(k, xk, xkm1) p_sys_noise(xk - sys(k, xkm1, 0));
%% Observation likelihood PDF p(y[k] | x[k])
% (under the suposition of additive process noise)
p_yk_given_xk = @(k, yk, xk) p_obs_noise(yk - obs(k, xk, 0));
%% Number of time steps
T = 10;
%% Separate memory space
x = zeros(nx,T);  y = zeros(ny,T);
u = zeros(nu,T);  v = zeros(nv,T);
%% Simulate system
xh0 = 0;                                  % initial state
u(:,1) = 0;                               % initial process noise
v(:,1) = gen_obs_noise(sigma_v);          % initial observation noise
x(:,1) = xh0;
y(:,1) = obs(1, xh0, v(:,1));
for k = 2:T
   % here we are basically sampling from p_xk_given_xkm1 and from p_yk_given_xk
   u(:,k) = gen_sys_noise();              % simulate process noise
   v(:,k) = gen_obs_noise();              % simulate observation noise
   x(:,k) = sys(k, x(:,k-1), u(:,k));     % simulate state
   y(:,k) = obs(k, x(:,k),   v(:,k));     % simulate observation
end
%% Separate memory
xh = zeros(nx, T); xh(:,1) = xh0;
yh = zeros(ny, T); yh(:,1) = obs(1, xh0, 0);
pf.k               = 1;                   % initial iteration number
pf.Ns              = 200;                 % number of particles
pf.w               = zeros(pf.Ns, T);     % weights
pf.particles       = zeros(nx, pf.Ns, T); % particles
pf.gen_x0          = gen_x0;              % function for sampling from initial pdf p_x0
pf.p_yk_given_xk   = p_yk_given_xk;       % function of the observation likelihood PDF p(y[k] | x[k])
pf.gen_sys_noise   = gen_sys_noise;       % function for generating system noise
%pf.p_x0 = p_x0;                          % initial prior PDF p(x[0])
%pf.p_xk_given_ xkm1 = p_xk_given_xkm1;   % transition prior PDF p(x[k] | x[k-1])
%% Estimate state
for k = 2:T
   fprintf('Iteration = %d/%d\n',k,T);
   % state estimation
   pf.k = k;
   %[xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'multinomial_resampling');
   [xh(:,k), pf] = particle_filter(sys, y(:,k), pf, 'systematic_resampling');   
 
   % filtered observation
   yh(:,k) = obs(k, xh(:,k), 0);
end
%% Make plots of the evolution of the density
figure
hold on;
xi = 1:T;
yi = -25:0.25:25;
[xx,yy] = meshgrid(xi,yi);
den = zeros(size(xx));
xhmode = zeros(size(xh));
for i = xi
   % for each time step perform a kernel density estimation
   den(:,i) = ksdensity(pf.particles(1,:,i), yi,'kernel','epanechnikov');
   [~, idx] = max(den(:,i));
   % estimate the mode of the density
   xhmode(i) = yi(idx);
   plot3(repmat(xi(i),length(yi),1), yi', den(:,i));
end
view(3);
box on;
title('Evolution of the state density','FontSize',14)
xlabel('Time/Iteration')
ylabel('State value')
zlabel('Probability')
figure
mesh(xx,yy,den);   
title('Evolution of the state density','FontSize',14)
xlabel('Time/Iteration')
ylabel('State value')
zlabel('Probability')
%% plot of the state vs estimated state by the particle filter vs particle paths
figure
hold on;
h1 = plot(1:T,squeeze(pf.particles),'y');
h2 = plot(1:T,x,'b','LineWidth',4);
h3 = plot(1:T,xh,'r','LineWidth',4);
h4 = plot(1:T,xhmode,'g','LineWidth',4);
legend([h2 h3 h4 h1(1)],'Simulated state','mean of estimated state','mode of estimated state','particle paths');
title('State vs estimated state by the particle filter vs particle paths','FontSize',14);
xlabel('Time/Iteration')
ylabel('State value')
%% plot of the observation vs filtered observation by the particle filter
figure
plot(1:T,y,'b', 1:T,yh,'r');
legend('observation','filtered observation');
title('Observation vs filtered observation by the particle filter','FontSize',14);
xlabel('Time/Iteration')
ylabel('State value')
return;
%%
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'Particles5.gif';
h2 = plot(1:T,x,'y','LineWidth',4);
title('Particle paths','FontSize',14);
xlabel('Time/Iteration')
ylabel('State value')
hold on
for n = 1:T
    m = ones(1,200)';
    scatter(n*m,pf.particles(:,:,n),'ro','MarkerFaceColor','b')
    pause(1)
    %h1 = scatterplot(pf.particles(n,:));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
      
end
%%
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'Particles6.gif';
xlim([1,10])
ylim([-20,15])

plot(1:T,squeeze(pf.particles),'y');
title('Particle paths','FontSize',14);
xlabel('Time/Iteration')
ylabel('State value')
hold on
for n = 1:T
    m = ones(1,200)';
    scatter(n*m,pf.particles(:,:,n),'ro','MarkerFaceColor','b')
    pause(1)
    %h1 = scatterplot(pf.particles(n,:));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
      
end
%%
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'pdf.gif';
plot3(repmat(xi(1),length(yi),1), yi', den(:,1));
title('Evolution of the state density','FontSize',14)
xlabel('Time/Iteration')
ylabel('State value')
zlabel('Probability')
hold on
for n = 2:T
    x = ones(1,200)';
    pause(1)
   plot3(repmat(xi(n),length(yi),1), yi', den(:,n));
    %h1 = scatterplot(pf.particles(n,:));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 2
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end
%%
%{
load("E:\0UNSW2_UG\ThesisB\HMM_system\RECOLA features\RECOLAR_all.mat")
%%
figure
plot(arousal_train{1}(1:7501,2))
title("Arousal ratings")
ylabel("Arousal rating")
xlabel("Frame/40ms")
%%
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'arousal.gif';
plot(arousal_train{1}(1:7501,2))
ylim([-1,1])
title("Arousal ratings")
ylabel("Arousal rating")
xlabel("Frame/40ms")
pause(2)
clf(h)
plot(arousal_train{1}(1:300,2))
title("Arousal ratings")
ylabel("Arousal rating")
xlabel("Frame/40ms")
hold on
for n = 1:300
    scatter(1*n,arousal_train{1}(n,2),'ro','MarkerFaceColor','b')
    %h1 = scatterplot(pf.particles(n,:));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end

%%
h = figure;
axis tight manual % this ensures that getframe() returns a consistent size
filename = 'arousal2.gif';
plot(arousal_train{1}(1:7501,2:7))
ylim([-1,1])
title("Arousal ratings")
ylabel("Arousal rating")
xlabel("Frame/40ms")
pause(2)
clf(h)
f = plot(arousal_train{1}(1:300,2:7))
c = get(f,'Color')
c{1}
c{2}
title("Arousal ratings")
ylabel("Arousal rating")
xlabel("Frame/40ms")
hold on
for n = 1:300
    scatter(1*n,arousal_train{1}(n,2),'ro','MarkerFaceColor',c{1})
    scatter(1*n,arousal_train{1}(n,3),'ro','MarkerFaceColor',c{2})
    scatter(1*n,arousal_train{1}(n,4),'ro','MarkerFaceColor',c{3})
    scatter(1*n,arousal_train{1}(n,5),'ro','MarkerFaceColor',c{4})
    scatter(1*n,arousal_train{1}(n,6),'ro','MarkerFaceColor',c{5})
    scatter(1*n,arousal_train{1}(n,7),'ro','MarkerFaceColor',c{6})
    %h1 = scatterplot(pf.particles(n,:));
    drawnow 
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append'); 
      end 
end
%%
load handel.mat

filename = 'handel.wav';
audiowrite(filename,y,Fs);
clear y Fs
%Read the data back into MATLAB using audioread.

[y,Fs] = audioread('handel.wav');
figure
plot((1:length(y))/Fs,y)
figure
subplot(2,1,2)
plot((1:length(y))/Fs,y(1:end))
title("Speech signal")
xlabel("Time(s)")
subplot(2,1,1)
plot((1:225)*0.04,arousal_dev{1}(1:225,2:7))
title("Arousal ratings")
xlabel("Time(s)")
ylabel("Arousal rating")
}%