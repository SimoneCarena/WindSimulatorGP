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

K = place(A,B,[-10,-11,-12,-13]);
Kp = K(:,1:2);
Kd = K(:,3:4);

%% Square Trajectory

wayPoints = [
   0.5 0.5 3.5 3.5 0.5;
   0.5 3.5 3.5 0.5 0.5
];

numSamples = 3000;

[q,qd,qdd,tvec,pp] = trapveltraj(wayPoints,numSamples,EndTime=30);

x0 = [
    q(1,1);
    q(2,1);
    qd(1,1);
    qd(2,1)
];

out = sim('trajectory_traking.slx');
x = out.x;
y = out.y;
vx = out.vx;
vy = out.vy;

figure
hold on
plot(x(1:numSamples),y(1:numSamples))
plot(q(1,:),q(2,:),'--')
legend('Object Position','Trajectory to Track')
title('Tracking')

figure
hold on
plot(tvec,x(1:numSamples))
plot(tvec,q(1,:))
title('x Position tracking performance')

figure
hold on
plot(tvec,y(1:numSamples))
plot(tvec,q(2,:))
title('y Position tracking performance')

% figure
% hold on
% plot(tvec,vx(1:numSamples))
% plot(tvec,qd(1,:))
% title('Vx Speed tracking performance')
% 
% figure
% hold on
% plot(tvec,vy(1:numSamples))
% plot(tvec,qd(2,:))
% title('Vy Speed tracking performance')

% Write Trajectory

save('../trajectories/square.mat','q','-v4')


%% Circular Trajectory

r = 1.5;
c = [2,2];
wayPoints = [];
i=1;
for theta=linspace(0,2*pi,100)
    wayPoints(1,i) = c(1)+r*cos(theta);
    wayPoints(2,i) = c(2)+r*sin(theta);
    i=i+1;
end

numSamples = 3000;
[q,qd,qdd,tvec,pp] = trapveltraj(wayPoints,numSamples);
x0 = [
    q(1,1);
    q(2,1);
    qd(1,1);
    qd(2,1)
];

out = sim('trajectory_traking.slx');
x = out.x;
y = out.y;
vx = out.vx;
vy = out.vy;

figure
hold on
plot(x(1:numSamples),y(1:numSamples))
plot(q(1,:),q(2,:),'--')
legend('Object Position','Trajectory to Track')
title('Tracking')

figure
hold on
plot(tvec,x(1:numSamples))
plot(tvec,q(1,:))
title('x Position tracking performance')

figure
hold on
plot(tvec,y(1:numSamples))
plot(tvec,q(2,:))
title('y Position tracking performance')

save('../trajectories/circle.mat','q','-v4')


%% Lemniscate Trajectory

a = 1.5;
c = [2;2];
wayPoints = [];
i=1;
R = [
    cos(pi/4) -sin(pi/4);
    sin(pi/4) cos(pi/4)
];
for theta=linspace(0,2*pi,100)
    p = [
        a*cos(theta)/(1+sin(theta)^2);
        a*sin(theta)*cos(theta)/(1+sin(theta)^2)
    ];
    p = R*p+c;
    wayPoints(:,i) = p;
    i=i+1;
end

numSamples = 3000;
[q,qd,qdd,tvec,pp] = trapveltraj(wayPoints,numSamples);
x0 = [
    q(1,1);
    q(2,1);
    qd(1,1);
    qd(2,1)
];

out = sim('trajectory_traking.slx');
x = out.x;
y = out.y;
vx = out.vx;
vy = out.vy;

figure
hold on
plot(x(1:numSamples),y(1:numSamples))
plot(q(1,:),q(2,:),'--')
legend('Object Position','Trajectory to Track')
title('Tracking')

figure
hold on
plot(tvec,x(1:numSamples))
plot(tvec,q(1,:))
title('x Position tracking performance')

figure
hold on
plot(tvec,y(1:numSamples))
plot(tvec,q(2,:))
title('y Position tracking performance')

save('../trajectories/lemniscate.mat','q','-v4')

%% Lemniscate Trajectory (2)

a = 1.5;
c = [2;2];
wayPoints = [];
i=1;
R = [
    cos(-pi/4) -sin(-pi/4);
    sin(-pi/4) cos(-pi/4)
];
for theta=linspace(0,2*pi,100)
    p = [
        a*cos(theta)/(1+sin(theta)^2);
        a*sin(theta)*cos(theta)/(1+sin(theta)^2)
    ];
    p = R*p+c;
    wayPoints(:,i) = p;
    i=i+1;
end

numSamples = 3000;
[q,qd,qdd,tvec,pp] = trapveltraj(wayPoints,numSamples);
x0 = [
    q(1,1);
    q(2,1);
    qd(1,1);
    qd(2,1)
];

out = sim('trajectory_traking.slx');
x = out.x;
y = out.y;
vx = out.vx;
vy = out.vy;

figure
hold on
plot(x(1:numSamples),y(1:numSamples))
plot(q(1,:),q(2,:),'--')
legend('Object Position','Trajectory to Track')
title('Tracking')

figure
hold on
plot(tvec,x(1:numSamples))
plot(tvec,q(1,:))
title('x Position tracking performance')

figure
hold on
plot(tvec,y(1:numSamples))
plot(tvec,q(2,:))
title('y Position tracking performance')

save('../trajectories/lemniscate2.mat','q','-v4')


%% Diamond Trajectory

wayPoints = [
   0.5 2.0 3.5 2.0 0.5;
   2.0 3.5 2.0 0.5 2.0
];

numSamples = 3000;

[q,qd,qdd,tvec,pp] = trapveltraj(wayPoints,numSamples,EndTime=30);

x0 = [
    q(1,1);
    q(2,1);
    qd(1,1);
    qd(2,1)
];

out = sim('trajectory_traking.slx');
x = out.x;
y = out.y;
vx = out.vx;
vy = out.vy;

figure
hold on
plot(x(1:numSamples),y(1:numSamples))
plot(q(1,:),q(2,:),'--')
legend('Object Position','Trajectory to Track')
title('Tracking')

figure
hold on
plot(tvec,x(1:numSamples))
plot(tvec,q(1,:))
title('x Position tracking performance')

figure
hold on
plot(tvec,y(1:numSamples))
plot(tvec,q(2,:))
title('y Position tracking performance')

% figure
% hold on
% plot(tvec,vx(1:numSamples))
% plot(tvec,qd(1,:))
% title('Vx Speed tracking performance')
% 
% figure
% hold on
% plot(tvec,vy(1:numSamples))
% plot(tvec,qd(2,:))
% title('Vy Speed tracking performance')

% Write Trajectory

save('../trajectories/diamond.mat','q','-v4')



