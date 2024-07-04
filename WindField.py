import numpy as np
import matplotlib.pyplot as plt
import json
import torch

from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from matplotlib.patches import Ellipse

from modules.Fan import Fan
from modules.System import System
from modules.Trajectory import Trajectory
from modules.PD import PD
from utils.function_parser import parse_generator
from utils.obstacle_parser import parse_obstacles
from utils.exceptions import MissingTrajectoryException, NoModelException

class WindField:
    '''
    Class used to model the wind field and simulate the evolution of the system moving in it.\\
    The Wind Field is constructed by passing it the wind field configuration file, and the 
    mass configuration file.
    '''
    def __init__(self, wind_field_conf_file, mass_conf_file, gp_predictor_x=None, gp_predictor_y=None):
        self.__wind_field_conf_file = wind_field_conf_file
        self.__mass_config_file = mass_conf_file
        self.__trajectory = None
        self.__gp_predictor_x = gp_predictor_x
        self.__gp_predictor_y = gp_predictor_y
        if gp_predictor_x is not None and gp_predictor_y is not None:
            self.__gp_predictor_x.eval()
            self.__gp_predictor_y.eval()

        self.__setup_wind_field(wind_field_conf_file)
        self.__setup_system(mass_conf_file)
        self.__setup_gp()
        self.__setup_plots()

    def __setup_wind_field(self, wind_field_conf_file):

        # Wind field configuration file
        file = open(wind_field_conf_file)
        data = json.load(file)

        ## Parse wind field and simulation data
        self.__width = data["width"]
        self.__height = data["height"]
        self.__duration = data["duration"] # Number of steps taken
        self.__dt = data["dt"] # Sampling time
        self.__air_density = data["air_density"] 
        self.__grid_resolution = data["grid_resolution"]
        self.__control_frequency = data["control_frequency"]
        obstacles_data = data["obstacles"]
        self.__obstacles = parse_obstacles(obstacles_data)

        ## Parse fans' data
        self.fans = []
        for fan in data["fans"]:
            x0 = float(fan["x0"])
            y0 = float(fan["y0"])
            alpha = np.deg2rad(float(fan["alpha"]))
            noise_var = float(fan['noise_var'])
            length = float(fan["length"])
            generator = fan["generator"]
            generator_function = parse_generator(generator)

            u0 = np.array([1,0])
            rot_mat = np.array([
                [np.cos(alpha),-np.sin(alpha)],
                [np.sin(alpha),np.cos(alpha)]
            ],dtype=float)
            u0 = rot_mat@u0

            f = Fan(x0,y0,u0[0],u0[1],length,noise_var,generator_function,self.__obstacles)
            self.fans.append(f)
        file.close()

    def __setup_system(self, mass_conf_file):
        # Parse system data
        file = open(mass_conf_file)
        data = json.load(file)
        m = data["m"]
        r = data["r"]

        # Actual system moving in the wind-field
        self.__system = System(m,r,self.__dt)

        # Controller Matrices
        Kp = np.diag(data["Kp"])
        Kd = np.diag(data["Kd"])
        # The controller's parameter were retrieved using MATLAB
        self.__pd = PD(Kp,Kd)
        # Setup controller saturation
        self.__lower_control_lim = data["u_min"]
        self.__upper_control_lim = data["u_max"]
        
    def __setup_gp(self):
        # Create arrays to train the GP model
        self.__gp_label_x = []
        self.__gp_label_y = []
        self.__gp_data = []

    def __setup_plots(self):
        # Create Plots arrays
        self.__xs = [] # List of x positions
        self.__ys = [] # List of y positions
        self.__vxs = [] # List of x velocities
        self.__vys = [] # List of y velocities
        self.__ctl_forces_x = [] # List of x control forces
        self.__ctl_forces_y = [] # List of y control forces
        self.__wind_force_x = [] # List of x wind forces
        self.__wind_force_y = [] # List of y wind forces
        self.__ex = [] # List of x position traking errors
        self.__ey = [] # List of y position traking errors
        self.__evx = [] # List of x velocity traking errors
        self.__evy = [] # List of y velocity traking errors

    def __draw_wind_field_grid(self,t=0.0):
        vxs = []
        vys = []
        vs = []
        for x in np.linspace(0.1,self.__width-0.1,self.__grid_resolution):
            vx = []
            vy = []
            v = []
            for y in np.linspace(0.1,self.__height-0.1,self.__grid_resolution):
                total_speed = np.zeros((2,),dtype=float)
                for fan in self.fans:
                    total_speed+=fan.generate_wind(x,y,t)
                vx.append(total_speed[0])
                vy.append(total_speed[1])
                v.append(np.linalg.norm(total_speed))
            vxs.append(vx)
            vys.append(vy)
            vs.append(v)
            t+=self.__dt
        return np.linspace(0.1,self.__width-0.1,self.__grid_resolution),np.linspace(0.1,self.__height-0.1,self.__grid_resolution),np.array(vxs),np.array(vys), np.array(vs)

    def plot(self, show=False, save=None):
        '''
        Plots the data related to the previosly run simulation.\\
        If the save parameter is set to `True`, the files are stored in the
        `imgs/trajectories_plots` folder
        '''
        if not self.__xs:
            print('No data to plot!')
            return

        T = [t*self.__dt for t in range(self.__duration)]
        p,v = self.__trajectory.trajectory()
        file_name = Path(self.__trajectory_name).stem
        sys_tr = np.stack([self.__xs,self.__ys])
        rmse = np.sqrt(1/len(p)*np.linalg.norm(sys_tr-p)**2)

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__xs,label='Object Position')
        ax[0].plot(T,p[0,:],'--',label='Reference Position')
        ax[0].title.set_text(r'Position ($x$)')
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$x$ $[m]$')
        ax[1].plot(T,self.__ex)
        ax[1].title.set_text(r'Traking error ($e_x$)')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_x$ $[m]$')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-x-position.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-x-position.svg')

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__ys,label='Object Position')
        ax[0].plot(T,p[1,:],'--',label='Reference Position')
        ax[0].title.set_text(r'Position ($y$)')
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$y$ $[m]$')
        ax[1].plot(T,self.__ey)
        ax[1].title.set_text(r'Traking error ($e_y$)')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_y$ $[m]$')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-y-position.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-y-position.svg')

        fig, ax = plt.subplots()
        ax.plot(np.NaN, np.NaN, '-', color='none', label='RMSE={:.2f} m'.format(rmse))
        ax.plot(self.__xs,self.__ys,label='System Trajectory')
        ax.plot(p[0,:],p[1,:],'--',label='Trajectory to Track')
        ax.title.set_text(r'Trajectory')
        ax.legend()
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-traking.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-traking.svg')

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__vxs)
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$V_x$ $[m/s]$')
        ax[0].title.set_text(r'Velocity ($V_x$)')
        ax[1].plot(T,self.__vys)
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$V_y$ $[m/s]$')
        ax[1].title.set_text(r'Velocity ($V_y$)')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-velocity.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-velocity.svg')

        fig, ax = plt.subplots(1,2)
        ax[0].plot(np.array(self.__idx_control)*self.__dt,self.__ctl_forces_x)
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$u_x$ $[N]$')
        ax[0].title.set_text(r'Control Force ($u_x$)')
        ax[1].plot(np.array(self.__idx_control)*self.__dt,self.__ctl_forces_y)
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$u_y$ $[N]$')
        ax[1].title.set_text(r'Control Force ($u_y$)')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-control-force.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-control-force.svg')

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__wind_force_x)
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$F_{wx}$ $[N]$')
        ax[0].title.set_text(r'Wind Force ($F_{wx}$)')
        ax[1].plot(T,self.__wind_force_y)
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$F_{wy}$ $[N]$')
        ax[1].title.set_text(r'Wind Force ($F_{wy}$)')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-wind-force.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-wind-force.svg')

        if show:
            plt.show()

        plt.close('all')

    def set_trajectory(self, trajectory_file,trajectory_name,laps=1):
        # Generate Trajectory
        self.__trajectory = Trajectory(trajectory_file,laps)
        self.__duration*=laps
        self.__trajectory_name = trajectory_name
        self.__tr_p, self.__tr_v = self.__trajectory.trajectory()

    def simulate_wind_field(self): 
        '''
        Runs the wind simulation. The wind field should be reset every time a new simulation.
        In case a GP model is being trained, the GP data should not be reset, as it stacks the subsequent
        measurements which can be used for training.
        '''
        if self.__trajectory is None:
            raise MissingTrajectoryException()

        # Set the mass initial conditions
        p,v = self.__trajectory.trajectory()
        x0 = p[0,0]
        y0 = p[1,0]
        self.__system.p[0] = x0
        self.__system.p[1] = y0
        self.__idx_control = []

        # Simulate the field 
        t = 0
        control_force = np.zeros((2,1))
        for target_p, target_v in self.__trajectory:
            total_speed = np.array([0,0],dtype=float)
            for fan in self.fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1],t)
                total_speed+=speed

            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            ep = target_p - self.__system.p
            ev = target_v - self.__system.v
            if t%self.__control_frequency == 0:
                self.__idx_control.append(t)
                # Collect inputs for GP
                self.__gp_data.append([self.__system.p[0],self.__system.p[1]])
                
                # Generate control force
                control_force = self.__pd.step(ep,ev)
                
                self.__ctl_forces_x.append(control_force[0])
                self.__ctl_forces_y.append(control_force[1])

                # Collect labels for GP
                self.__gp_label_x.append(wind_force[0])
                self.__gp_label_y.append(wind_force[1])

            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__xs.append(self.__system.p[0])
            self.__ys.append(self.__system.p[1])
            self.__vxs.append(self.__system.v[0])
            self.__vys.append(self.__system.v[1])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            # Simulate Dynamics
            self.__system.discrete_dynamics(wind_force+control_force)
            t+=1

    def simulate_gp(self, max_size, predictors, p0=None, show=False, save=None, kernel_name=''):
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        predictor_x = predictors[0]
        predictor_y = predictors[1]
        
        x_pred = []
        y_pred = []
        x_lower = []
        x_upper = []
        y_lower = []
        y_upper = []
        predicted_x_pos = []
        predicted_y_pos = []
        idxs = []
        covs = []

        # Set the mass initial conditions
        p,_ = self.__trajectory.trajectory()
        x0 = p[0,0]
        y0 = p[1,0]
        self.__system.p[0] = x0
        self.__system.p[1] = y0

        dummy = System(self.__system.m,self.__system.r,self.__system.dt*self.__control_frequency)
        dummy.p[0] = x0
        dummy.p[1] = y0

        # Simulate the field 
        t = 0
        k = 0
        control_force = np.zeros((2,1))
        for target_p, target_v in self.__trajectory:
            total_speed = np.array([0,0],dtype=float)
            for fan in self.fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            ep = target_p - self.__system.p
            ev = target_v - self.__system.v
            self.__xs.append(self.__system.p[0])
            self.__ys.append(self.__system.p[1])
            self.__vxs.append(self.__system.v[0])
            self.__vys.append(self.__system.v[1])
            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            if t%self.__control_frequency==0:

                # Collect inputs for GP
                self.__gp_data.append([self.__system.p[0],self.__system.p[1]])
            
                # Generate control force
                control_force = self.__pd.step(ep,ev)
                # Include force in the computation
                p = torch.FloatTensor([[self.__system.p[0],self.__system.p[1]]])
                if k >= max_size:
                    idxs.append(t)
                    predicted_wind_force_x = predictor_x(p)
                    predicted_wind_force_y = predictor_y(p)
                    control_force -= np.array([
                        predicted_wind_force_x.mean.item(),
                        predicted_wind_force_y.mean.item()
                    ])
                    # Collect Data For Plot
                    cov_x = predicted_wind_force_x.covariance_matrix.item()
                    cov_y = predicted_wind_force_y.covariance_matrix.item()
                    covs.append(np.diag([cov_x,cov_y]))
                    x_pred.append(predicted_wind_force_x.mean.item())
                    y_pred.append(predicted_wind_force_y.mean.item())
                    lower, upper = predicted_wind_force_x.confidence_region()
                    x_lower.append(lower.item())
                    x_upper.append(upper.item())
                    lower, upper = predicted_wind_force_y.confidence_region()
                    y_lower.append(lower.item())
                    y_upper.append(upper.item())
                    dummy.discrete_dynamics(control_force)
                    predicted_x_pos.append(dummy.p[0])
                    predicted_y_pos.append(dummy.p[1])

                    # Collect labels for GP
                    self.__gp_label_x.append(wind_force[0])
                    self.__gp_label_y.append(wind_force[1])

                # Update GP Model
                if k==0:
                    predictor_x.set_train_data(p,torch.FloatTensor([wind_force[0]]),strict=False)
                    predictor_y.set_train_data(p,torch.FloatTensor([wind_force[1]]),strict=False)
                elif k>=max_size:
                    gp_data = predictor_x.train_inputs[0]
                    gp_labels = predictor_x.train_targets
                    predictor_x.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force[0]])]),strict=False)
                    gp_data = predictor_y.train_inputs[0]
                    gp_labels = predictor_y.train_targets
                    predictor_y.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force[1]])]),strict=False)
                elif k<max_size:
                    gp_data = predictor_x.train_inputs[0]
                    gp_labels = predictor_x.train_targets
                    predictor_x.set_train_data(torch.cat([gp_data,p],dim=0),torch.cat([gp_labels,torch.FloatTensor([wind_force[0]])],dim=0),strict=False)
                    gp_data = predictor_y.train_inputs[0]
                    gp_labels = predictor_y.train_targets
                    predictor_y.set_train_data(torch.cat([gp_data,p],dim=0),torch.cat([gp_labels,torch.FloatTensor([wind_force[1]])],dim=0),strict=False)
                k+=1
            
            control_force = control_force.clip(self.__lower_control_lim,self.__upper_control_lim)
            self.__ctl_forces_x.append(control_force[0])
            self.__ctl_forces_y.append(control_force[1])

            # Simulate Dynamics
            self.__system.discrete_dynamics(wind_force+control_force)
            dummy.set_state(self.__system.p.copy(),self.__system.v.copy())

            t+=1

        # Plots
        T = np.array([t*self.__dt for t in range(self.__duration)])
        self.__wind_force_x = np.array(self.__wind_force_x)
        self.__wind_force_y = np.array(self.__wind_force_y)
        self.__xs = np.array(self.__xs)
        self.__ys = np.array(self.__ys)

        pos_var = np.array(covs)*(self.__dt*self.__control_frequency)**2
        eigs = []
        eigenvectors = []
        for m in pos_var:
            eigs.append(np.linalg.eigvals(m))
            eigenvectors.append(np.linalg.eig(m)[1])
        angles = []
        for e in eigenvectors:
            angles.append(np.arctan2(e[1, 0], e[0, 0]))
        confidence_intervals = [
            [np.sqrt(5.991*eigs[i][0]),np.sqrt(5.991*eigs[i][1])] for i in range(len(eigs))
        ]
        pos_lower = np.array([[
            predicted_x_pos[i]-confidence_intervals[i][0],
            predicted_y_pos[i]-confidence_intervals[i][1],
        ] for i in range(len(confidence_intervals))])
        pos_upper = np.array([[
            predicted_x_pos[i]+confidence_intervals[i][0],
            predicted_y_pos[i]+confidence_intervals[i][1],
        ] for i in range(len(confidence_intervals))])
        ellipses = []
        for i in range(len(predicted_x_pos)):
            ellipses.append(
                Ellipse((predicted_x_pos[i],predicted_y_pos[i]),
                        confidence_intervals[i][0],
                        confidence_intervals[i][1],
                        angle=angles[i],
                        facecolor='cyan',
                        alpha=0.5)
            )

        # # Plot x prediction
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(f'One Step-Ahead Prediction (x-axis) {self.__trajectory_name} Trajectory with {kernel_name} Kernel')
        fig.tight_layout(pad=3.0)
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__wind_force_x,'--',color='orange',label='Real Wind Force')
        ax[0].plot(T[idxs],x_pred,'b-',label="estimated Wind Force")
        ax[0].fill_between(T[idxs], x_lower, x_upper, alpha=0.5, color='cyan',label='Confidence')
        # ax[0].plot(T[idxs],self.__wind_force_x[idxs],'g*',label="Sampled Data")
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$F_{wx}$ $[N]$')
        ax[0].title.set_text('GP Wind Prediction')
        ## Plot Estimation Error
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(T[idxs],self.__wind_force_x[idxs]-np.array(x_pred),label='Prediction Error')
        ax[1].legend()
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{F_{wx}}$ $[N]$')
        ax[1].title.set_text('GP Prediction Error')

        if save is not None:
            fig.savefig(save+f'wind-x-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'wind-x-{self.__trajectory_name}-{kernel_name}.svg')

        # Plot y prediction
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(f'One Step-Ahead Prediction (y-axis) {self.__trajectory_name} Trajectory with {kernel_name} Kernel')
        fig.tight_layout(pad=3.0)
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__wind_force_y,'--',color='orange',label='Real Wind Force')
        ax[0].plot(T[idxs],y_pred,'b-',label="estimated Wind Force")
        ax[0].fill_between(T[idxs], y_lower, y_upper, alpha=0.5, color='cyan',label='Confidence')
        # ax[0].plot(T[idxs],self.__wind_force_y[idxs],'g*',label="Sampled Data")
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$F_{wy}$ $[N]$')
        ax[0].title.set_text('GP Wind Prediction')
        ## Plot Estimation Error
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(T[idxs],self.__wind_force_y[idxs]-np.array(y_pred),label='Prediction Error')
        ax[1].legend()
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{F_{wy}}$ $[N]$')
        ax[1].title.set_text('GP Prediction Error')

        if save is not None:
            fig.savefig(save+f'wind-y-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'wind-y-{self.__trajectory_name}-{kernel_name}.svg')

        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(r'$x$-Position Prediction')
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__xs,'--',color='orange',label=r'Real $x$ Position')
        ax[0].plot(T[idxs],predicted_x_pos,'b',label=r'Prediction $x$',alpha=0.5)
        ax[0].fill_between(T[idxs], pos_lower[:,0], pos_upper[:,0], alpha=0.5, color='cyan',label='Confidence')
        ax[0].legend()
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(np.array(self.__xs[idxs])-np.array(predicted_x_pos),label="Prediction Error")
        ax[1].legend()

        if save is not None:
            fig.savefig(save+f'position-x-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'position-x-{self.__trajectory_name}-{kernel_name}.svg')

        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(r'$y$-Position Prediction')
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__ys,'--',color='orange',label=r'Real $y$ Position')
        ax[0].plot(T[idxs],predicted_y_pos,'b',label=r'Prediction $y$',alpha=0.5)
        ax[0].fill_between(T[idxs], pos_lower[:,1], pos_upper[:,1], alpha=0.5, color='cyan',label='Confidence')
        ax[0].legend()
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(np.array(self.__ys[idxs])-np.array(predicted_y_pos),label="Prediction Error")
        ax[1].legend()

        if save is not None:
            fig.savefig(save+f'position-y-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'position-y-{self.__trajectory_name}-{kernel_name}.svg')

        tr, _ = self.__trajectory.trajectory()
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.suptitle('Real vs Reference Trajectory')
        fig.tight_layout()
        xs, ys, vx, vy, v = self.__draw_wind_field_grid()
        v_max = np.max(v)
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        for i in range(len(xs)):
            for j in range(len(ys)):
                ax.arrow(xs[i],ys[j],vx[i,j]/100,vy[i,j]/100,length_includes_head=False,head_width=0.015,head_length=0.015,width=0.005,color='orange',alpha=v[i,j]/v_max)
        # Create custom colormap
        colors = [(1, 0.5, 0, alpha) for alpha in np.linspace(0, 1, 256)]
        orange_transparency_cmap = LinearSegmentedColormap.from_list('orange_transparency', colors, N=256)
        bar = ax.imshow(np.array([[0,v_max]]), cmap=orange_transparency_cmap)
        bar.set_visible(False)
        cb = fig.colorbar(bar,orientation="vertical")
        cb.set_label(label=r'Wind Speed $[m/s]$',labelpad=10)
        ax.plot(self.__xs,self.__ys,'b',label='System Trajectory')
        ax.plot(tr[0,:],tr[1,:],'--',color='chartreuse',label='Reference Trajectory')
        ax.plot(self.__xs[0],self.__ys[0],'bo',markersize=5,label='Starting Position')
        ax.legend()

        if save is not None:
            fig.savefig(save+f'system-trajectory-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'system-trajectory-{self.__trajectory_name}-{kernel_name}.svg')

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        fig.suptitle('Predicted vs System Trajectory')
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        ax.plot(predicted_x_pos,predicted_y_pos,'b-',label='Predicted Trajectory')
        ax.plot(self.__xs,self.__ys,'--',color='orange',label='System Trajectory')
        for ellipse in ellipses:
            ax.add_patch(ellipse)
        ax.plot(np.nan,color='cyan',alpha=0.5,label='Confidence')
        ax.legend()
        if save is not None:
            fig.savefig(save+f'estimated-trajectory-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'estimated-trajectory-{self.__trajectory_name}-{kernel_name}.svg')

        if show:
            plt.show()
        plt.close('all')

        # self.animate(
        #     # f'imgs/animations/{kernel_name}-{self.__trajectory_name}-trajectory.gif'
        # )

    def simulate_mogp(self, max_size, predictor, p0=None, show=False, save=None, kernel_name=''):
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        x_pred = []
        y_pred = []
        x_lower = []
        x_upper = []
        y_lower = []
        y_upper = []
        predicted_x_pos = []
        predicted_y_pos = []
        idxs = []
        covs = []

        # Set the mass initial conditions
        p,_ = self.__trajectory.trajectory()
        if p0 is None:
            x0 = p[0,0]
            y0 = p[1,0]
        else:
            x0 = p0[0]
            y0 = p0[1]
        self.__system.p[0] = x0
        self.__system.p[1] = y0

        dummy = System(self.__system.m,self.__system.r,self.__system.dt*self.__control_frequency)
        dummy.p[0] = x0
        dummy.p[1] = y0

        # Simulate the field 
        t = 0
        k = 0
        control_force = np.zeros((2,1))
        for target_p, target_v in self.__trajectory:
            total_speed = np.array([0,0],dtype=float)
            for fan in self.fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            ep = target_p - self.__system.p
            ev = target_v - self.__system.v
            self.__xs.append(self.__system.p[0])
            self.__ys.append(self.__system.p[1])
            self.__vxs.append(self.__system.v[0])
            self.__vys.append(self.__system.v[1])
            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            if t%self.__control_frequency==0:

                # Collect inputs for GP
                self.__gp_data.append([self.__system.p[0],self.__system.p[1]])
            
                # Generate control force
                control_force = self.__pd.step(ep,ev)
                # Include force in the computation
                p = torch.FloatTensor([[self.__system.p[0],self.__system.p[1]]])
                if k >= max_size:
                    idxs.append(t)
                    predicted_wind_force = predictor(p)
                    control_force -= predicted_wind_force.mean[0].numpy()
                    # Collect Data For Plot
                    covs.append(predicted_wind_force.covariance_matrix)
                    x_pred.append(predicted_wind_force.mean[0,0].item())
                    y_pred.append(predicted_wind_force.mean[0,1].item())
                    lower, upper = predicted_wind_force.confidence_region()
                    x_lower.append(lower[0,0].item())
                    x_upper.append(upper[0,0].item())
                    y_lower.append(lower[0,1].item())
                    y_upper.append(upper[0,1].item())
                    dummy.discrete_dynamics(control_force)
                    predicted_x_pos.append(dummy.p[0])
                    predicted_y_pos.append(dummy.p[1])

                    # Collect labels for GP
                    self.__gp_label_x.append(wind_force[0])
                    self.__gp_label_y.append(wind_force[1])

                # Update GP Model
                if k==0:
                    predictor.set_train_data(p,torch.FloatTensor([wind_force]),strict=False)
                elif k>=max_size:
                    gp_data = predictor.train_inputs[0]
                    gp_labels = predictor.train_targets
                    predictor.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force])]),strict=False)
                elif k<max_size:
                    gp_data = predictor.train_inputs[0]
                    gp_labels = predictor.train_targets
                    predictor.set_train_data(torch.cat([gp_data,p],dim=0),torch.cat([gp_labels,torch.FloatTensor([wind_force])],dim=0),strict=False)
                k+=1
            
            control_force = control_force.clip(self.__lower_control_lim,self.__upper_control_lim)
            self.__ctl_forces_x.append(control_force[0])
            self.__ctl_forces_y.append(control_force[1])

            # Simulate Dynamics
            self.__system.discrete_dynamics(wind_force+control_force)
            dummy.set_state(self.__system.p.copy(),self.__system.v.copy())

            t+=1

        # Plots
        T = np.array([t*self.__dt for t in range(self.__duration)])
        self.__wind_force_x = np.array(self.__wind_force_x)
        self.__wind_force_y = np.array(self.__wind_force_y)
        self.__xs = np.array(self.__xs)
        self.__ys = np.array(self.__ys)

        pos_var = np.array(covs)*(self.__dt*self.__control_frequency)**2
        eigs = []
        eigenvectors = []
        for m in pos_var:
            eigs.append(np.linalg.eigvals(m))
            eigenvectors.append(np.linalg.eig(m)[1])
        angles = []
        for e in eigenvectors:
            angles.append(np.arctan2(e[1, 0], e[0, 0]))
        confidence_intervals = [
            [np.sqrt(5.991*eigs[i][0]),np.sqrt(5.991*eigs[i][1])] for i in range(len(eigs))
        ]
        pos_lower = np.array([[
            predicted_x_pos[i]-confidence_intervals[i][0],
            predicted_y_pos[i]-confidence_intervals[i][1],
        ] for i in range(len(confidence_intervals))])
        pos_upper = np.array([[
            predicted_x_pos[i]+confidence_intervals[i][0],
            predicted_y_pos[i]+confidence_intervals[i][1],
        ] for i in range(len(confidence_intervals))])
        ellipses = []
        for i in range(len(predicted_x_pos)):
            ellipses.append(
                Ellipse((predicted_x_pos[i],predicted_y_pos[i]),
                        confidence_intervals[i][0],
                        confidence_intervals[i][1],
                        angle=angles[i],
                        facecolor='cyan',
                        alpha=0.5)
            )

        # # Plot x prediction
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(f'One Step-Ahead Prediction (x-axis) {self.__trajectory_name} Trajectory with {kernel_name} Kernel')
        fig.tight_layout(pad=3.0)
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__wind_force_x,'--',color='orange',label='Real Wind Force')
        ax[0].plot(T[idxs],x_pred,'b-',label="estimated Wind Force")
        ax[0].fill_between(T[idxs], x_lower, x_upper, alpha=0.5, color='cyan',label='Confidence')
        # ax[0].plot(T[idxs],self.__wind_force_x[idxs],'g*',label="Sampled Data")
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$F_{wx}$ $[N]$')
        ax[0].title.set_text('GP Wind Prediction')
        ## Plot Estimation Error
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(T[idxs],self.__wind_force_x[idxs]-np.array(x_pred),label='Prediction Error')
        ax[1].legend()
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{F_{wx}}$ $[N]$')
        ax[1].title.set_text('GP Prediction Error')

        if save is not None:
            fig.savefig(save+f'wind-x-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'wind-x-{self.__trajectory_name}-{kernel_name}.svg')

        # Plot y prediction
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(f'One Step-Ahead Prediction (y-axis) {self.__trajectory_name} Trajectory with {kernel_name} Kernel')
        fig.tight_layout(pad=3.0)
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__wind_force_y,'--',color='orange',label='Real Wind Force')
        ax[0].plot(T[idxs],y_pred,'b-',label="estimated Wind Force")
        ax[0].fill_between(T[idxs], y_lower, y_upper, alpha=0.5, color='cyan',label='Confidence')
        # ax[0].plot(T[idxs],self.__wind_force_y[idxs],'g*',label="Sampled Data")
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$F_{wy}$ $[N]$')
        ax[0].title.set_text('GP Wind Prediction')
        ## Plot Estimation Error
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(T[idxs],self.__wind_force_y[idxs]-np.array(y_pred),label='Prediction Error')
        ax[1].legend()
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{F_{wy}}$ $[N]$')
        ax[1].title.set_text('GP Prediction Error')

        if save is not None:
            fig.savefig(save+f'wind-y-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'wind-y-{self.__trajectory_name}-{kernel_name}.svg')

        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(r'$x$-Position Prediction')
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__xs,'--',color='orange',label=r'Real $x$ Position')
        ax[0].plot(T[idxs],predicted_x_pos,'b',label=r'Prediction $x$',alpha=0.5)
        ax[0].fill_between(T[idxs], pos_lower[:,0], pos_upper[:,0], alpha=0.5, color='cyan',label='Confidence')
        ax[0].legend()
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(np.array(self.__xs[idxs])-np.array(predicted_x_pos),label="Prediction Error")
        ax[1].legend()

        if save is not None:
            fig.savefig(save+f'position-x-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'position-x-{self.__trajectory_name}-{kernel_name}.svg')

        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.suptitle(r'$y$-Position Prediction')
        ax[0].set_xlim([0,T[-1]])
        ax[0].plot(T,self.__ys,'--',color='orange',label=r'Real $y$ Position')
        ax[0].plot(T[idxs],predicted_y_pos,'b',label=r'Prediction $y$',alpha=0.5)
        ax[0].fill_between(T[idxs], pos_lower[:,1], pos_upper[:,1], alpha=0.5, color='cyan',label='Confidence')
        ax[0].legend()
        ax[1].set_xlim([0,T[-1]])
        ax[1].plot(np.array(self.__ys[idxs])-np.array(predicted_y_pos),label="Prediction Error")
        ax[1].legend()

        if save is not None:
            fig.savefig(save+f'position-y-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'position-y-{self.__trajectory_name}-{kernel_name}.svg')

        tr, _ = self.__trajectory.trajectory()
        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.suptitle('Real vs Reference Trajectory')
        fig.tight_layout()
        xs, ys, vx, vy, v = self.__draw_wind_field_grid()
        v_max = np.max(v)
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        for i in range(len(xs)):
            for j in range(len(ys)):
                ax.arrow(xs[i],ys[j],vx[i,j]/100,vy[i,j]/100,length_includes_head=False,head_width=0.015,head_length=0.015,width=0.005,color='orange',alpha=v[i,j]/v_max)
        # Create custom colormap
        colors = [(1, 0.5, 0, alpha) for alpha in np.linspace(0, 1, 256)]
        orange_transparency_cmap = LinearSegmentedColormap.from_list('orange_transparency', colors, N=256)
        bar = ax.imshow(np.array([[0,v_max]]), cmap=orange_transparency_cmap)
        bar.set_visible(False)
        cb = fig.colorbar(bar,orientation="vertical")
        cb.set_label(label=r'Wind Speed $[m/s]$',labelpad=10)
        ax.plot(self.__xs,self.__ys,'b',label='System Trajectory')
        ax.plot(tr[0,:],tr[1,:],'--',color='chartreuse',label='Reference Trajectory')
        ax.plot(self.__xs[0],self.__ys[0],'bo',markersize=5,label='Starting Position')
        ax.legend()

        if save is not None:
            fig.savefig(save+f'system-trajectory-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'system-trajectory-{self.__trajectory_name}-{kernel_name}.svg')

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        fig.suptitle('Predicted vs System Trajectory')
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        ax.plot(predicted_x_pos,predicted_y_pos,'b-',label='Predicted Trajectory')
        ax.plot(self.__xs,self.__ys,'--',color='orange',label='System Trajectory')
        for ellipse in ellipses:
            ax.add_patch(ellipse)
        ax.plot(np.nan,color='cyan',alpha=0.5,label='Confidence')
        ax.axis('equal')
        ax.legend()
        if save is not None:
            fig.savefig(save+f'estimated-trajectory-{self.__trajectory_name}-{kernel_name}.png')
            fig.savefig(save+f'estimated-trajectory-{self.__trajectory_name}-{kernel_name}.svg')

        if show:
            plt.show()
        plt.close('all')

        self.animate(
            # f'imgs/animations/{kernel_name}-{self.__trajectory_name}-trajectory.gif'
        )

    def reset(self, wind_field_conf_file=None, mass_conf_file=None, gp_predictor_x=None, gp_predictor_y=None):
        '''
        Resets the Wind Field, based on the files that are passes, if something is omitted,
        the prviously used configuration file will be considered.\\
        This should be executed before every new test is run.\\
        This does not reset the GP files.
        '''
        if wind_field_conf_file is not None:
            self.__setup_wind_field(wind_field_conf_file)
            self.__wind_field_conf_file = wind_field_conf_file
        else:
            self.__setup_wind_field(self.__wind_field_conf_file)
        
        if mass_conf_file is not None:
            self.__setup_system(mass_conf_file)
            self.__mass_config_file = mass_conf_file
        else:
            self.__setup_system(self.__mass_config_file)

        if gp_predictor_x is not None:
            self.__gp_predictor_x = gp_predictor_x
        if gp_predictor_y is not None:
            self.__gp_predictor_y = gp_predictor_y

        self.__setup_plots()

    def get_gp_data(self):
        '''
        Returns the GP data needed for training or testing.\\
        The data is in the form (x,y), Fx, Fy, T
        '''
        return self.__gp_data.copy(), self.__gp_label_x.copy(), self.__gp_label_y.copy()
    
    def get_wind_field_data(self):
        
        inputs = np.array([
            (self.__xs[i],self.__ys[i]) for i in range(len(self.__xs))
        ])
        x_labels = self.__wind_force_x.copy()
        y_labels = self.__wind_force_y.copy()

        return inputs, x_labels, y_labels, 
    
    def reset_gp(self):
        '''
        Resets the GP data
        '''
        self.__setup_gp()

    def animate(self, save=None, scale=25, interval=10):
        '''
        Plots the animation showing the evolution of the system following the trajectory
        in the wind field
        '''
        
        file_name = self.__trajectory_name

        fig, ax = plt.subplots()
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        def animation_function(t):
            t = int(t*scale)
            ax.clear()
            ax.set_xlim([0,self.__width])
            ax.set_ylim([0,self.__height])
            ax.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*self.__dt))
            ax.plot(self.__tr_p[0,t],self.__tr_p[1,t],'o',color='orange',markersize=7,label='Target Distance=[{0:.2f},{0:.2f}] m'.format(self.__ex[t],self.__ey[t])) # Traget Location
            ax.plot(self.__xs[t],self.__ys[t],'bo',markersize=5) # Object Moving
            ax.quiver(self.__xs[t],self.__ys[t],self.__wind_force_x[t],self.__wind_force_y[t],scale=75,width=0.003,color='r',label='Wind Force=[{0:.2f},{0:.2f}] N'.format(self.__wind_force_x[t],self.__wind_force_y[t])) # Wind Force
            ax.quiver(self.__xs[t],self.__ys[t],self.__ctl_forces_x[t],self.__ctl_forces_y[t],scale=75,width=0.003,color="#4DBEEE",label='Control Force=[{0:.2f},{0:.2f}] N'.format(self.__ctl_forces_x[t],self.__ctl_forces_y[t])) # Control Force
            ax.plot(self.__xs[:t],self.__ys[:t],'b')
            ax.plot(self.__tr_p[0,:t],self.__tr_p[1,:t],'--',color='orange')
            ax.legend()

        anim = animation.FuncAnimation(fig,animation_function,frames=int(self.__duration/scale),interval=interval,repeat=False)

        if save is not None:
            anim.save(save,writer=animation.FFMpegWriter(fps=30))

        plt.show()
        plt.close('all')

    def draw_wind_field(self,show=False,save=None):

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)

        if self.__trajectory is not None:
            tr, _ = self.__trajectory.trajectory()
            ax.plot(tr[0,:],tr[1,:],'c')

        xs, ys, vx, vy, v = self.__draw_wind_field_grid()
        v_max = np.max(v)
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        for i in range(len(xs)):
            for j in range(len(ys)):
                ax.arrow(xs[i],ys[j],vx[i,j]/(v_max*10),vy[i,j]/(v_max*10),length_includes_head=False,head_width=0.015,head_length=0.015,width=0.005,color='orange',alpha=v[i,j]/v_max)

        # Create custom colormap
        colors = [(1, 0.5, 0, alpha) for alpha in np.linspace(0, 1, 256)]
        orange_transparency_cmap = LinearSegmentedColormap.from_list('orange_transparency', colors, N=256)
        bar = ax.imshow(np.array([[0,v_max]]), cmap=orange_transparency_cmap)
        bar.set_visible(False)
        cb = fig.colorbar(bar,orientation="vertical")
        cb.set_label(label=r'Wind Speed $[m/s]$',labelpad=10)
        for o in self.__obstacles:
            circle = plt.Circle((o.x,o.y),o.r,color='k')
            ax.add_patch(circle)
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        ax.set_title('Wind Field')
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        fig.legend(['Trajectory','Wind Speed'])
        if save is not None:
            plt.savefig(save+f'/wind-field.png',dpi=300)
            plt.savefig(save+f'/wind-field.svg')

        if show:    
            plt.show()
        
        plt.close('all')