close all
clear all
clc

%% Model's Parameters

tau_phi = 0.18;
tau_theta = 0.18;
tau_psi = 0.56;
tau_a = 0.05;
k = 1;

v0 = [0;0;0];
p0 = [0;0;0];
attitude_0 = [pi/2;pi/2;pi/2;1];

g = 9.81;

Kp = 20*eye(4,2);
Kd = 10*eye(4,2);

%% Simulation 

p_ref = [2;2];
v_ref = [0;0];
















