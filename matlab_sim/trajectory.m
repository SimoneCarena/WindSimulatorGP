close all
clear all
clc

%% System Description

m = 1;

A = [
    0 0 1 0;
    0 0 0 1;
    0 0 0 0;
    0 0 0 0;
];

B = [
    0 0;
    0 0;
    1/m 0;
    0 1/m
];

C = eye(4);
D = zeros(4,2);

K = place(A,B,[-5,-6,-7,-8]);
Kp = K(:,1:2);
Kd = K(:,3:4);

%% Ellipse Trajectory

c = [2;2];
a = 1;
b = 1.5;
theta_0 = 0;
theta_f = 2*pi;
T = 30;
duration = 30000;

[q,qd,qdd,tvec,pp] = trapveltraj([theta_0,theta_f],duration,EndTime=T);

syms theta(t)
x = a*cos(theta);
y = b*sin(theta);
xd = diff(x,t);
yd = diff(y,t);

X = double(subs(x,theta(t),q))+c(1);
Y = double(subs(y,theta(t),q))+c(2);
XD = double(subs(xd,{diff(theta(t),t) theta(t)},{qd q}));
YD = double(subs(yd,{diff(theta(t),t) theta(t)},{qd q}));

data = [X;Y;XD;YD];
save('../trajectories/ellipse.mat','data','-v4')

%% Plots

figure
plot(tvec,XD)
xlabel('t [s]')
ylabel('Vx [m/s]')

figure
plot(tvec,YD)
xlabel('t [s]')
ylabel('Vy [m/s]')

figure
plot(X,Y)
xlabel('x [m]')
ylabel('y [m]')


%% Simulation

x0 = a*cos(0)+c(1);
y0 = b*sin(0)+c(1);
X_ref = timeseries(X,tvec);
Y_ref = timeseries(Y,tvec);
XD_ref = timeseries(XD,tvec);
YD_ref = timeseries(YD,tvec);

out = sim('trajectory_traking.slx');

x_sim = get(out,'x_sim');
y_sim = get(out,'y_sim');
xd_sim = get(out,'xd_sim');
yd_sim = get(out,'yd_sim');

%% Simulation Plots

figure
hold on
plot(x_sim,y_sim,'b-')
plot(X,Y,'--')
xlabel('x [m]')
ylabel('y [m]')
legend('System Trajectory','Reference Trajectory')

figure
hold on
plot(tvec,xd_sim,'b-')
plot(tvec,XD,'--')
xlabel('t [s]')
ylabel('Vx [m/s]')
legend('System x Velocity','Reference x Velocity')

figure
hold on
plot(tvec,yd_sim,'b-')
plot(tvec,YD,'--')
xlabel('t [s]')
ylabel('Vy [m/s]')
legend('System y Velocity','Reference y Velocity')

figure
hold on
plot(tvec,x_sim,'b-')
plot(tvec,X,'--')
xlabel('t [s]')
ylabel('x [m]')
legend('System x Position','Reference x Position')

figure
hold on
plot(tvec,y_sim,'b-')
plot(tvec,Y,'--')
xlabel('t [s]')
ylabel('y [m]')
legend('System y Position','Reference y Position')


%% Lemniscate Trajectory

c = [2;2];
a = 1.5;
theta_0 = 0;
theta_f = 2*pi;
T = 30;
duration = 30000;

[q,qd,qdd,tvec,pp] = trapveltraj([theta_0,theta_f],duration,EndTime=T);

syms theta(t)
x = a*cos(theta)/(1+sin(theta)^2);
y = a*sin(theta)*cos(theta)/(1+sin(theta)^2);
xd = diff(x,t);
yd = diff(y,t);

X = double(subs(x,theta(t),q))+c(1);
Y = double(subs(y,theta(t),q))+c(2);
XD = double(subs(xd,{diff(theta(t),t) theta(t)},{qd q}));
YD = double(subs(yd,{diff(theta(t),t) theta(t)},{qd q}));

data = [X;Y;XD;YD];
save('../trajectories/lemniscate.mat','data','-v4')

%% Plots

figure
plot(tvec,XD)
xlabel('t [s]')
ylabel('Vx [m/s]')

figure
plot(tvec,YD)
xlabel('t [s]')
ylabel('Vy [m/s]')

figure
plot(X,Y)
xlabel('x [m]')
ylabel('y [m]')
axis equal

%% Simulation

x0 = X(1);
y0 = Y(1);

X_ref = timeseries(X,tvec);
Y_ref = timeseries(Y,tvec);
XD_ref = timeseries(XD,tvec);
YD_ref = timeseries(YD,tvec);

out = sim('trajectory_traking.slx');

x_sim = get(out,'x_sim');
y_sim = get(out,'y_sim');
xd_sim = get(out,'xd_sim');
yd_sim = get(out,'yd_sim');

%% Simulation Plots

figure
hold on
axis equal
plot(x_sim,y_sim,'b-')
plot(X,Y,'--')
xlabel('x [m]')
ylabel('y [m]')
legend('System Trajectory','Reference Trajectory')

figure
hold on
plot(tvec,xd_sim,'b-')
plot(tvec,XD,'--')
xlabel('t [s]')
ylabel('Vx [m/s]')
legend('System x Velocity','Reference x Velocity')

figure
hold on
plot(tvec,yd_sim,'b-')
plot(tvec,YD,'--')
xlabel('t [s]')
ylabel('Vy [m/s]')
legend('System y Velocity','Reference y Velocity')

figure
hold on
plot(tvec,x_sim,'b-')
plot(tvec,X,'--')
xlabel('t [s]')
ylabel('x [m]')
legend('System x Position','Reference x Position')

figure
hold on
plot(tvec,y_sim,'b-')
plot(tvec,Y,'--')
xlabel('t [s]')
ylabel('y [m]')
legend('System y Position','Reference y Position')

%% Lemniscate2 Trajectory

c = [2;2];
a = 1.5;
theta_0 = 0;
theta_f = 2*pi;
T = 30;
duration = 30000;

[q,qd,qdd,tvec,pp] = trapveltraj([theta_0,theta_f],duration,EndTime=T);

syms theta(t)
x = a*sin(theta)*cos(theta)/(1+sin(theta)^2);
y = a*cos(theta)/(1+sin(theta)^2);
xd = diff(x,t);
yd = diff(y,t);

X = double(subs(x,theta(t),q))+c(1);
Y = double(subs(y,theta(t),q))+c(2);
XD = double(subs(xd,{diff(theta(t),t) theta(t)},{qd q}));
YD = double(subs(yd,{diff(theta(t),t) theta(t)},{qd q}));

data = [X;Y;XD;YD];
save('../trajectories/lemniscate2.mat','data','-v4')

%% Plots

figure
plot(tvec,XD)
xlabel('t [s]')
ylabel('Vx [m/s]')

figure
plot(tvec,YD)
xlabel('t [s]')
ylabel('Vy [m/s]')

figure
plot(X,Y)
xlabel('x [m]')
ylabel('y [m]')
axis equal

%% Simulation

x0 = X(1);
y0 = Y(1);

X_ref = timeseries(X,tvec);
Y_ref = timeseries(Y,tvec);
XD_ref = timeseries(XD,tvec);
YD_ref = timeseries(YD,tvec);

out = sim('trajectory_traking.slx');

x_sim = get(out,'x_sim');
y_sim = get(out,'y_sim');
xd_sim = get(out,'xd_sim');
yd_sim = get(out,'yd_sim');

%% Simulation Plots

figure
hold on
axis equal
plot(x_sim,y_sim,'b-')
plot(X,Y,'--')
xlabel('x [m]')
ylabel('y [m]')
legend('System Trajectory','Reference Trajectory')

figure
hold on
plot(tvec,xd_sim,'b-')
plot(tvec,XD,'--')
xlabel('t [s]')
ylabel('Vx [m/s]')
legend('System x Velocity','Reference x Velocity')

figure
hold on
plot(tvec,yd_sim,'b-')
plot(tvec,YD,'--')
xlabel('t [s]')
ylabel('Vy [m/s]')
legend('System y Velocity','Reference y Velocity')

figure
hold on
plot(tvec,x_sim,'b-')
plot(tvec,X,'--')
xlabel('t [s]')
ylabel('x [m]')
legend('System x Position','Reference x Position')

figure
hold on
plot(tvec,y_sim,'b-')
plot(tvec,Y,'--')
xlabel('t [s]')
ylabel('y [m]')
legend('System y Position','Reference y Position')

%% Lemniscate3 Trajectory

c = [2;2];
a = 1.5;
theta_0 = 0;
theta_f = 2*pi;
T = 30;
duration = 30000;
alpha = pi/4;

[q,qd,qdd,tvec,pp] = trapveltraj([theta_0,theta_f],duration,EndTime=T);

syms theta(t)
x_ = a*cos(theta)/(1+sin(theta)^2);
y_ = a*sin(theta)*cos(theta)/(1+sin(theta)^2);
x = x_*cos(alpha)-y_*sin(alpha);
y = x_*cos(alpha)+y_*sin(alpha);
xd = diff(x,t);
yd = diff(y,t);

X = double(subs(x,theta(t),q))+c(1);
Y = double(subs(y,theta(t),q))+c(2);
XD = double(subs(xd,{diff(theta(t),t) theta(t)},{qd q}));
YD = double(subs(yd,{diff(theta(t),t) theta(t)},{qd q}));

data = [X;Y;XD;YD];
save('../trajectories/lemniscate3.mat','data','-v4')

%% Plots

figure
plot(tvec,XD)
xlabel('t [s]')
ylabel('Vx [m/s]')

figure
plot(tvec,YD)
xlabel('t [s]')
ylabel('Vy [m/s]')

figure
plot(X,Y)
xlabel('x [m]')
ylabel('y [m]')
axis equal

%% Simulation

x0 = X(1);
y0 = Y(1);

X_ref = timeseries(X,tvec);
Y_ref = timeseries(Y,tvec);
XD_ref = timeseries(XD,tvec);
YD_ref = timeseries(YD,tvec);

out = sim('trajectory_traking.slx');

x_sim = get(out,'x_sim');
y_sim = get(out,'y_sim');
xd_sim = get(out,'xd_sim');
yd_sim = get(out,'yd_sim');

%% Simulation Plots

figure
hold on
axis equal
plot(x_sim,y_sim,'b-')
plot(X,Y,'--')
xlabel('x [m]')
ylabel('y [m]')
legend('System Trajectory','Reference Trajectory')

figure
hold on
plot(tvec,xd_sim,'b-')
plot(tvec,XD,'--')
xlabel('t [s]')
ylabel('Vx [m/s]')
legend('System x Velocity','Reference x Velocity')

figure
hold on
plot(tvec,yd_sim,'b-')
plot(tvec,YD,'--')
xlabel('t [s]')
ylabel('Vy [m/s]')
legend('System y Velocity','Reference y Velocity')

figure
hold on
plot(tvec,x_sim,'b-')
plot(tvec,X,'--')
xlabel('t [s]')
ylabel('x [m]')
legend('System x Position','Reference x Position')

figure
hold on
plot(tvec,y_sim,'b-')
plot(tvec,Y,'--')
xlabel('t [s]')
ylabel('y [m]')
legend('System y Position','Reference y Position')


%% Ellipse2 Trajectory

c = [2;2];
a = 1;
b = 1.5;
theta_0 = 0;
theta_f = 2*pi;
T = 30;
duration = 30000;
alpha = -pi/4;

[q,qd,qdd,tvec,pp] = trapveltraj([theta_0,theta_f],duration,EndTime=T);

syms theta(t)
x_ = a*cos(theta);
y_ = b*sin(theta);
x = x_*cos(alpha)-y_*sin(alpha);
y = x_*cos(alpha)+y_*sin(alpha);
xd = diff(x,t);
yd = diff(y,t);

X = double(subs(x,theta(t),q))+c(1);
Y = double(subs(y,theta(t),q))+c(2);
XD = double(subs(xd,{diff(theta(t),t) theta(t)},{qd q}));
YD = double(subs(yd,{diff(theta(t),t) theta(t)},{qd q}));

data = [X;Y;XD;YD];
save('../trajectories/ellipse2.mat','data','-v4')

%% Plots

figure
plot(tvec,XD)
xlabel('t [s]')
ylabel('Vx [m/s]')

figure
plot(tvec,YD)
xlabel('t [s]')
ylabel('Vy [m/s]')

figure
plot(X,Y)
xlabel('x [m]')
ylabel('y [m]')


%% Simulation

x0 = a*cos(0)+c(1);
y0 = b*sin(0)+c(2);
X_ref = timeseries(X,tvec);
Y_ref = timeseries(Y,tvec);
XD_ref = timeseries(XD,tvec);
YD_ref = timeseries(YD,tvec);

out = sim('trajectory_traking.slx');

x_sim = get(out,'x_sim');
y_sim = get(out,'y_sim');
xd_sim = get(out,'xd_sim');
yd_sim = get(out,'yd_sim');

%% Simulation Plots

figure
hold on
axis equal
plot(x_sim,y_sim,'b-')
plot(X,Y,'--')
xlabel('x [m]')
ylabel('y [m]')
legend('System Trajectory','Reference Trajectory')

figure
hold on
plot(tvec,xd_sim,'b-')
plot(tvec,XD,'--')
xlabel('t [s]')
ylabel('Vx [m/s]')
legend('System x Velocity','Reference x Velocity')

figure
hold on
plot(tvec,yd_sim,'b-')
plot(tvec,YD,'--')
xlabel('t [s]')
ylabel('Vy [m/s]')
legend('System y Velocity','Reference y Velocity')

figure
hold on
plot(tvec,x_sim,'b-')
plot(tvec,X,'--')
xlabel('t [s]')
ylabel('x [m]')
legend('System x Position','Reference x Position')

figure
hold on
plot(tvec,y_sim,'b-')
plot(tvec,Y,'--')
xlabel('t [s]')
ylabel('y [m]')
legend('System y Position','Reference y Position')

%% Lemniscate4 Trajectory

c = [2;2];
a = 1.5;
theta_0 = 0;
theta_f = 2*pi;
T = 30;
duration = 30000;
alpha = -pi/4;

[q,qd,qdd,tvec,pp] = trapveltraj([theta_0,theta_f],duration,EndTime=T);

syms theta(t)
x_ = a*cos(theta)/(1+sin(theta)^2);
y_ = a*sin(theta)*cos(theta)/(1+sin(theta)^2);
x = x_*cos(alpha)-y_*sin(alpha);
y = x_*sin(alpha)+y_*cos(alpha);
xd = diff(x,t);
yd = diff(y,t);

X = double(subs(x,theta(t),q))+c(1);
Y = double(subs(y,theta(t),q))+c(2);
XD = double(subs(xd,{diff(theta(t),t) theta(t)},{qd q}));
YD = double(subs(yd,{diff(theta(t),t) theta(t)},{qd q}));

data = [X;Y;XD;YD];
save('../test_trajectories/lemniscate4.mat','data','-v4')

%% Plots

figure
plot(tvec,XD)
xlabel('t [s]')
ylabel('Vx [m/s]')

figure
plot(tvec,YD)
xlabel('t [s]')
ylabel('Vy [m/s]')

figure
plot(X,Y)
xlabel('x [m]')
ylabel('y [m]')
axis equal

%% Simulation

x0 = X(1);
y0 = Y(1);

X_ref = timeseries(X,tvec);
Y_ref = timeseries(Y,tvec);
XD_ref = timeseries(XD,tvec);
YD_ref = timeseries(YD,tvec);

out = sim('trajectory_traking.slx');

x_sim = get(out,'x_sim');
y_sim = get(out,'y_sim');
xd_sim = get(out,'xd_sim');
yd_sim = get(out,'yd_sim');

%% Simulation Plots

figure
hold on
plot(x_sim,y_sim,'b-')
plot(X,Y,'--')
xlabel('x [m]')
ylabel('y [m]')
legend('System Trajectory','Reference Trajectory')

figure
hold on
plot(tvec,xd_sim,'b-')
plot(tvec,XD,'--')
xlabel('t [s]')
ylabel('Vx [m/s]')
legend('System x Velocity','Reference x Velocity')

figure
hold on
plot(tvec,yd_sim,'b-')
plot(tvec,YD,'--')
xlabel('t [s]')
ylabel('Vy [m/s]')
legend('System y Velocity','Reference y Velocity')

figure
hold on
plot(tvec,x_sim,'b-')
plot(tvec,X,'--')
xlabel('t [s]')
ylabel('x [m]')
legend('System x Position','Reference x Position')

figure
hold on
plot(tvec,y_sim,'b-')
plot(tvec,Y,'--')
xlabel('t [s]')
ylabel('y [m]')
legend('System y Position','Reference y Position')
