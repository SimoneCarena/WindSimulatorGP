import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from modules.Fan import Fan
from modules.Quadrotor import Quadrotor
from modules.Trajectory import Trajectory
from modules.MPC import MPC
from modules.Kernels import RBFKernel
from modules.GPModel import GPModel
from utils.function_parser import parse_generator
from utils.obstacle_parser import parse_obstacles
from utils.exceptions import MissingTrajectoryException

class WindField:
    '''
    Class used to model the wind field and simulate the evolution of the system moving in it.\\
    The Wind Field is constructed by passing it the wind field configuration file, and the 
    mass configuration file.
    '''
    def __init__(self, wind_field_conf_file):
        self.__wind_field_conf_file = wind_field_conf_file
        self.__trajectory = None

        self.__setup_wind_field(wind_field_conf_file)
        self.__setup_system()
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

    def __setup_system(self):
        self.__quadrotor = Quadrotor(
            self.__dt,
            np.zeros(10)
        )
        self.__control_horizon = 10
        self.__mpc = MPC(
            self.__quadrotor.get_dynamics(),
            self.__control_horizon,
            self.__dt*self.__control_frequency,
            Q=100*np.eye(6), 
            R=0.1*np.eye(4),
            input_dim=2,
            output_dim=3,
            window_size=50,
            obstacles=self.__obstacles
        )       
        
    def __setup_gp(self):
        # Create arrays to train the GP model
        self.__gp_label_x = []
        self.__gp_label_y = []
        self.__gp_data = []

    def __setup_plots(self):
        # Create Plots arrays
        self.__xs = [] # List of x positions
        self.__ys = [] # List of y positions
        self.__zs = [] # List of z positions
        self.__vxs = [] # List of x velocities
        self.__vys = [] # List of y velocities
        self.__vzs = [] # List of z velocities
        self.__ctl_phi = [] # List of phi control 
        self.__ctl_theta = [] # List of theta control 
        self.__ctl_psi = [] # List of psi control 
        self.__ctl_a = [] # List of thrust control 
        self.__wind_force_x = [] # List of x wind forces
        self.__wind_force_y = [] # List of y wind forces
        self.__ex = [] # List of x position traking errors
        self.__ey = [] # List of y position traking errors
        self.__ez = [] # List of y position traking errors
        self.__evx = [] # List of x velocity traking errors
        self.__evy = [] # List of y velocity traking errors
        self.__evz = [] # List of z velocity traking errors

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
        T = [t*self.__dt for t in range(self.__duration)]
        p,v = self.__trajectory.trajectory()
        file_name = Path(self.__trajectory_name).stem
        sys_tr = np.stack([self.__xs,self.__ys,self.__zs])
        rmse = np.sqrt(1/len(self.__xs)*np.linalg.norm(sys_tr-p)**2)

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

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__zs,label='Object Position')
        ax[0].plot(T,p[2,:],'--',label='Reference Position')
        ax[0].title.set_text(r'Position ($z$)')
        ax[0].legend()
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$z$ $[m]$')
        ax[1].plot(T,self.__ez)
        ax[1].title.set_text(r'Traking error ($e_z$)')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_y$ $[m]$')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-z-position.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-z-position.svg')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        fig.suptitle('System Trajectory')
        ax.plot(self.__xs,self.__ys,self.__zs,'-',color='c',label='System Trajectory')
        ax.plot(p[0,:],p[1,:],p[2,:],'--',color='orange',label='Reference Trajectory')
        ax.plot(self.__xs[0],self.__ys[0],self.__zs[0],'bo',label='Starting Position')
        ax.plot(self.__xs[-1],self.__ys[-1],self.__zs[-1],'ro',label='End Position')
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        ax.set_zlabel(r'$z$ $[m]$')
        ax.set_xlim([0,4])
        ax.set_ylim([0,4])
        ax.set_zlim([0,4])
        ax.legend()
        plt.show()
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-traking.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-traking.svg')

        fig, ax = plt.subplots(3,1)
        ax[0].plot(T,self.__vxs,label='Quadrotor Speed')
        ax[0].plot(T,v[0,:],'--',label='Reference Speed')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$V_x$ $[m/s]$')
        ax[0].title.set_text(r'Velocity ($V_x$)')
        ax[1].plot(T,self.__vys,label='Quadrotor Speed')
        ax[1].plot(T,v[1,:],'--',label='Reference Speed')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$V_y$ $[m/s]$')
        ax[1].title.set_text(r'Velocity ($V_y$)')
        ax[2].plot(T,self.__vzs,label='Quadrotor Speed')
        ax[2].plot(T,v[2,:],'--',label='Reference Speed')
        ax[2].set_xlabel(r'$t$ $[s]$')
        ax[2].set_ylabel(r'$V_z$ $[m/s]$')
        ax[2].title.set_text(r'Velocity ($V_z$)')
        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-velocity.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-velocity.svg')

        fig, ax = plt.subplots(4,1)
        ax[0].plot(np.array(self.__idx_control)*self.__dt,self.__ctl_phi)
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$\phi^c$ $[red]$')
        ax[0].title.set_text(r'Control $\phi$ ($\phi^c$)')
        ax[1].plot(np.array(self.__idx_control)*self.__dt,self.__ctl_theta)
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$\theta^c$ $[red]$')
        ax[1].title.set_text(r'Control $\theta$ ($\theta^c$)')
        ax[2].plot(np.array(self.__idx_control)*self.__dt,self.__ctl_psi)
        ax[2].set_xlabel(r'$t$ $[s]$')
        ax[2].set_ylabel(r'$\psi^c$ $[red]$')
        ax[2].title.set_text(r'Control $\psi$ ($\psi^c$)')
        ax[3].plot(np.array(self.__idx_control)*self.__dt,self.__ctl_a)
        ax[3].set_xlabel(r'$t$ $[s]$')
        ax[3].set_ylabel(r'$a_c$ $[m/s^2]$')
        ax[3].title.set_text(r'Control Thrust ($a^c$)')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-control-action.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-control-action.svg')

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
        self.__trajectory = Trajectory(trajectory_file,laps,[2,0])
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
        target_p,target_v = self.__trajectory.trajectory()
        self.__quadrotor.set_state(np.array([
            target_p[0,0], target_p[1,0], 0,
            target_v[0,0], target_v[1,0], target_v[2,0],
            0,0,0,0
        ])
        )
        self.__idx_control = []

        # Simulate the field 
        control_force = np.zeros((4,1))
        print(f'Simulating {self.__trajectory_name} Trajectory...')
        for t in range(len(self.__trajectory)):
            print(
                '|{}{}| {:.2f}% ({:.2f}/{:.2f} s)'.format(
                    '█'*int(20*(t+1)/len(target_p[0,:])),
                    ' '*(20-int(20*(t+1)/len(target_p[0,:]))),
                    (t+1)/len(target_p[0,:])*100,
                    (t+1)*self.__dt,
                    len(target_p[0,:])*self.__dt
                ),
                end='\r'
            )
            total_speed = np.array([0,0],dtype=float)
            state = self.__quadrotor.get_state()
            for fan in self.fans:
                speed = fan.generate_wind(state[0],state[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = 0.1*total_speed*np.sign(total_speed)
            ep = target_p[:,t] - state[:3]
            ev = target_v[:,t] - state[3:6]
            if t%self.__control_frequency == 0:
                self.__idx_control.append(t)
                # Collect inputs for GP
                state = self.__quadrotor.get_state()
                self.__gp_data.append([state[0],state[1]])
                # Generate MPC Reference
                idx = min(t+self.__control_horizon*self.__control_frequency,len(self.__trajectory))
                ref = np.concatenate([
                        target_p[:,t:idx:self.__control_frequency],
                        target_v[:,t:idx:self.__control_frequency]
                ],axis=0)
                # If the remaining trajectory is < than the control horizon
                # expand it using the last refence
                if (idx-t)//self.__control_frequency < self.__control_horizon:
                    ref = np.concatenate([
                        ref,
                        np.repeat(ref[:,-1,np.newaxis],self.__control_horizon-(idx-t)//self.__control_frequency,axis=1)
                    ],axis=1)
                # Generate control force
                control_force, _ = self.__mpc(state,ref)
                
                self.__ctl_phi.append(control_force[0])
                self.__ctl_theta.append(control_force[1])
                self.__ctl_psi.append(control_force[2])
                self.__ctl_a.append(control_force[3])

                # Collect labels for GP
                self.__gp_label_x.append(wind_force[0])
                self.__gp_label_y.append(wind_force[1])

            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__ez.append(ep[2])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__evz.append(ev[2])
            self.__xs.append(state[0])
            self.__ys.append(state[1])
            self.__zs.append(state[2])
            self.__vxs.append(state[3])
            self.__vys.append(state[4])
            self.__vzs.append(state[5])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            # Simulate Dynamics
            self.__quadrotor.step(control_force,np.hstack([wind_force,0]))
        
        print('')

    def simulate_gp(self, window_size, predictors, p0=None, show=False, save=None, kernel_name=''):
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        predictor_x = predictors[0]
        predictor_y = predictors[1]
        
        predicted_x_pos = []
        predicted_y_pos = []
        covs = []
        x_wind_map = []
        y_wind_map = []
        x_uncertainty_map = []
        y_uncertainty_map = []
        real_wind_x_map = []
        real_wind_y_map= []

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
        target_p, target_v = self.__trajectory.trajectory()
        for t in range(len(self.__trajectory)):
            total_speed = np.array([0,0],dtype=float)
            for fan in self.fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            ep = target_p[:,t] - self.__system.p
            ev = target_v[:,t] - self.__system.v
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
                if k > 0:
                    predicted_wind_force_x = predictor_x(p)
                    predicted_wind_force_y = predictor_y(p)
                    # Collect Data For Plot
                    cov_x = predicted_wind_force_x.covariance_matrix.item()
                    cov_y = predicted_wind_force_y.covariance_matrix.item()
                    covs.append(np.diag([cov_x,cov_y]))
                    dummy.discrete_dynamics(
                        np.clip(
                            control_force,
                            self.__lower_control_lim,
                            self.__upper_control_lim
                        )
                    )
                    control_force -= np.array([
                        predicted_wind_force_x.mean.item(),
                        predicted_wind_force_y.mean.item()
                    ])
                    predicted_x_pos.append(dummy.p[0])
                    predicted_y_pos.append(dummy.p[1])

                    # Collect labels for GP
                    self.__gp_label_x.append(wind_force[0])
                    self.__gp_label_y.append(wind_force[1])

                    # Generate Heatmap
                    current_x_wind_map = []
                    current_y_wind_map = []
                    current_x_unc_map = []
                    current_y_unc_map = []
                    for i in np.linspace(0,self.__width,self.__grid_resolution):
                        x_wind_pred = []
                        y_wind_pred = []
                        x_std_pred = []
                        y_std_pred = []
                        for j in np.linspace(0,self.__width,self.__grid_resolution):
                            heatmap_pred_x = predictor_x(torch.FloatTensor([[i,j]]))
                            heatmap_pred_y = predictor_y(torch.FloatTensor([[i,j]]))
                            x_wind_pred.append(heatmap_pred_x.mean.item())
                            y_wind_pred.append(heatmap_pred_y.mean.item())
                            x_std_pred.append(np.sqrt(
                                heatmap_pred_x.covariance_matrix.item()
                            ))
                            y_std_pred.append(np.sqrt(
                                heatmap_pred_y.covariance_matrix.item()
                            ))
                        current_x_wind_map.append(x_wind_pred)
                        current_y_wind_map.append(y_wind_pred)
                        current_x_unc_map.append(x_std_pred)
                        current_y_unc_map.append(y_std_pred)
                    x_wind_map.append(current_x_wind_map)
                    y_wind_map.append(current_y_wind_map)
                    x_uncertainty_map.append(current_x_unc_map)
                    y_uncertainty_map.append(current_y_unc_map)
                    _, _, real_vx, real_vy, _ = self.__draw_wind_field_grid(t*self.__dt)
                    real_fx = (0.5*self.__air_density*self.__system.surf)*real_vx**2*np.sign(real_vx)
                    real_fy = (0.5*self.__air_density*self.__system.surf)*real_vy**2*np.sign(real_vy)
                    real_wind_x_map.append(real_fx.copy())
                    real_wind_y_map.append(real_fy.copy())

                # Update GP Model
                if k==0:
                    predictor_x.set_train_data(p,torch.FloatTensor([wind_force[0]]),strict=False)
                    predictor_y.set_train_data(p,torch.FloatTensor([wind_force[1]]),strict=False)
                elif k>=window_size:
                    gp_data = predictor_x.train_inputs[0]
                    gp_labels = predictor_x.train_targets
                    predictor_x.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force[0]])]),strict=False)
                    gp_data = predictor_y.train_inputs[0]
                    gp_labels = predictor_y.train_targets
                    predictor_y.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force[1]])]),strict=False)
                elif k<window_size:
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


        # Setup Plots
        x_wind_map = np.array(x_wind_map)
        y_wind_map = np.array(y_wind_map)
        x_uncertainty_map = np.array(x_uncertainty_map)
        y_uncertainty_map = np.array(y_uncertainty_map)
        real_wind_x_map = np.array(real_wind_x_map)
        real_wind_y_map = np.array(real_wind_y_map)
        max_std_x = np.max(x_uncertainty_map)
        max_std_y = np.max(y_uncertainty_map)
        max_std = np.max(np.array([max_std_x,max_std_y]))
        min_std_x = np.min(x_uncertainty_map)
        min_std_y = np.min(y_uncertainty_map)
        min_std = np.min(np.array([min_std_x,min_std_y]))

        # Plots
        fig = plt.figure(figsize=(16, 9))
        outer = gridspec.GridSpec(1, 3, width_ratios = [0.48, 0.48, 0.04])
        ax1 = plt.Subplot(fig, outer[0])
        fig.add_subplot(ax1)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[1], wspace=0.1, hspace=0.4)
        ax21 = plt.Subplot(fig, inner[0])
        ax22 = plt.Subplot(fig, inner[1])
        fig.add_subplot(ax21)
        fig.add_subplot(ax22)
        norm = Normalize(vmin=min_std, vmax=max_std)
        sm = ScalarMappable(cmap='RdBu_r', norm=norm)
        sm.set_array([])
        cbar_ax = plt.subplot(outer[2])
        cbar = fig.colorbar(sm, cax=cbar_ax,ticks=np.linspace(min_std, max_std, 5),format='%.2f')
        cbar.set_label(r'Uncertainty $\sqrt{\sigma^2}$ $[N]$')
        fig.tight_layout(pad=5)

        # Create Grid
        x_grid, y_grid = np.meshgrid(np.linspace(0.0,self.__width,self.__grid_resolution),np.linspace(0.0,self.__height,self.__grid_resolution),indexing='ij')

        scale = self.__control_frequency
        def animation_function(t):
            # Clear figures and setup plots
            ax1.clear()
            ax21.clear()
            ax22.clear()
            ax1.set_xlabel(r'$x$ $[m]$')
            ax1.set_ylabel(r'$y$ $[m]$')
            ax21.set_xlabel(r'$x$ $[m]$')
            ax21.set_ylabel(r'$y$ $[m]$')
            ax22.set_xlabel(r'$x$ $[m]$')
            ax22.set_ylabel(r'$y$ $[m]$')

            t = int(t*scale)
            k = t//scale - 1
            start = max(0,k-window_size)*scale
            ax1.set_xlim([0.0,self.__width])
            ax1.set_ylim([0.0,self.__height])
            ax21.title.set_text('Uncertainty Heatmap x Prediction')
            ax22.title.set_text('Uncertainty Heatmap y Prediction')

            # Plot System Evolution
            ax1.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*self.__dt))
            ax1.plot(self.__xs[t],self.__ys[t],'bo',label='System\'s Position')
            ax1.plot(self.__xs[start:t:scale],self.__ys[start:t:scale],'o-',color=(0.878, 0.867, 0.137,0.5),markerfacecolor=(0.878, 0.867, 0.137,1.0),markeredgecolor=(0.878, 0.867, 0.137,1.0),linewidth=8,markersize=2,label='Active Window')
            
            # Plot Heatmaps
            if k>0:
                ax21.imshow(x_uncertainty_map[k],cmap='RdBu_r',vmin=min_std,vmax=max_std,extent=(0.0,4.0,0.0,4.0))
                ax22.imshow(y_uncertainty_map[k],cmap='RdBu_r',vmin=min_std,vmax=max_std,extent=(0.0,4.0,0.0,4.0))
                for i in range(self.__grid_resolution):
                    for j in range(self.__grid_resolution):
                        ax1.arrow(x_grid[i,j],y_grid[i,j],real_wind_x_map[k,i,j]/25,real_wind_y_map[k,i,j]/25,color='r',alpha=0.5,head_width=0.015,head_length=0.015,width=0.008)
                        ax1.arrow(x_grid[i,j],y_grid[i,j],x_wind_map[k,i,j]/25,y_wind_map[k,i,j]/25,color='c',alpha=0.5,head_width=0.015,head_length=0.015,width=0.008)
                ax1.plot(np.nan,np.nan,'>',color='r',alpha=0.5,label='Real Wind Force')
                ax1.plot(np.nan,np.nan,'>',color='c',alpha=0.5,label='Estimated Wind Force')

            ax1.legend()
            
        anim = animation.FuncAnimation(fig,animation_function,frames=int(self.__duration/scale),interval=100,repeat=True)

        if show:
            plt.show()
        if save:
            print('Rendering...')
            anim.save(f'imgs/animations/heatmap-{kernel_name}-{self.__trajectory_name}.gif',writer=animation.FFMpegWriter(fps=30))
        plt.close('all')

    def simulate_gp_horizon(self, window_size, predictors, horizon=1, p0=None, show=False, save=False, kernel_name=''):
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        predictor_x = predictors[0]
        predictor_y = predictors[1]
        
        predicted_X_pos = []
        predicted_Y_pos = []
        idxs = []
        Covs = []

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

        k = 0
        control_force = np.zeros((2,))
        target_p, target_v = self.__trajectory.trajectory()
        for t in range(len(self.__trajectory)):
            total_speed = np.array([0,0],dtype=float)
            for fan in self.fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            ep = target_p[:,t] - self.__system.p
            ev = target_v[:,t] - self.__system.v
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
                control_force = self.__pd.step(ep,ev)
                p = torch.FloatTensor([[self.__system.p[0],self.__system.p[1]]])
                # Simulate system's evolution within the horizon
                if k > 0:
                    idxs.append(t)
                    predicted_x_pos = []
                    predicted_y_pos = []
                    covs = []
                    for j in range(horizon):
                        idx = min(t+j*self.__control_frequency,len(self.__trajectory)-1)
                        ep = target_p[:,idx] - dummy.p
                        ev = target_v[:,idx] - dummy.v
                        dummy_control_force = self.__pd.step(ep,ev)
                        dummy_pos = torch.FloatTensor([[
                            dummy.p[0],
                            dummy.p[1]
                        ]])
                        predicted_wind_force_x = predictor_x(dummy_pos)
                        predicted_wind_force_y = predictor_y(dummy_pos)
                        # Collect Data For Plot
                        covs.append(np.diag([
                            predicted_wind_force_x.covariance_matrix.item(),
                            predicted_wind_force_y.covariance_matrix.item()
                        ]))
                        dummy_control_force = dummy_control_force.clip(self.__lower_control_lim,self.__upper_control_lim)
                        dummy.discrete_dynamics(dummy_control_force)
                        predicted_x_pos.append(dummy.p[0])
                        predicted_y_pos.append(dummy.p[1])
                    predicted_X_pos.append(predicted_x_pos)
                    predicted_Y_pos.append(predicted_y_pos)
                    Covs.append(covs)

                    # Generate control force for the real system
                    predicted_wind_force_x = predictor_x(p)
                    predicted_wind_force_y = predictor_y(p)
                    control_force -= np.array([
                        predicted_wind_force_x.mean.item(),
                        predicted_wind_force_y.mean.item()
                    ])

                # Update GP Model
                # Update GP Model
                if k==0:
                    predictor_x.set_train_data(p,torch.FloatTensor([wind_force[0]]),strict=False)
                    predictor_y.set_train_data(p,torch.FloatTensor([wind_force[1]]),strict=False)
                elif k>=window_size:
                    gp_data = predictor_x.train_inputs[0]
                    gp_labels = predictor_x.train_targets
                    predictor_x.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force[0]])]),strict=False)
                    gp_data = predictor_y.train_inputs[0]
                    gp_labels = predictor_y.train_targets
                    predictor_y.set_train_data(torch.cat([gp_data[1:,],p],dim=0),torch.cat([gp_labels[1:],torch.FloatTensor([wind_force[1]])]),strict=False)
                elif k<window_size:
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

        # Derive Uncertainties
        Pos_upper = []
        Pos_lower = []
        Ellipses = []
        for k in range(len(Covs)):
            covs = Covs[k]
            predicted_x_pos = predicted_X_pos[k]
            predicted_y_pos = predicted_Y_pos[k]
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

            Pos_upper.append(pos_upper)
            Pos_lower.append(pos_lower)
            Ellipses.append(ellipses)

        # Animation
        predicted_X_pos = np.array(predicted_X_pos)
        predicted_Y_pos = np.array(predicted_Y_pos)

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=4)
        fig.suptitle(f'{horizon}-Step Ahead Prediction using {kernel_name} Kernel')
        scale = self.__control_frequency # Plot frequency
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        # Create colorbar for wind speed
        _, _, _, _, v = self.__draw_wind_field_grid()
        v_max = np.max(v)
        colors = [(1, 0, 0, alpha) for alpha in np.linspace(0, 1, 256)]
        orange_transparency_cmap = LinearSegmentedColormap.from_list('orange_transparency', colors, N=256)
        bar = ax.imshow(np.array([[0,v_max]]), cmap=orange_transparency_cmap)
        bar.set_visible(False)
        cb = fig.colorbar(bar,orientation="vertical")
        cb.set_label(label=r'Wind Speed $[m/s]$',labelpad=10)
        def animation_function(t):
            ax.clear()
            t = int(t*scale)
            k = t//scale - 1
            start = max(0,k-window_size)*scale
            # Draw wind field
            xs, ys, vx, vy, v = self.__draw_wind_field_grid(t*self.__dt)
            for i in range(len(xs)):
                for j in range(len(ys)):
                    ax.arrow(xs[i],ys[j],vx[i,j]/100,vy[i,j]/100,length_includes_head=False,head_width=0.015,head_length=0.015,width=0.005,color='r',alpha=min(v[i,j]/v_max,1.0))
            ax.set_xlim([0.0,self.__width])
            ax.set_ylim([0.0,self.__height])
            ax.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*self.__dt))
            ax.plot(self.__xs[:t],self.__ys[:t],'b',label='System Trajectory')
            ax.plot(target_p[0,t],target_p[1,t],'o',color='orange')
            ax.plot(self.__xs[t],self.__ys[t],'bo')
            ax.plot(target_p[0,:t],target_p[1,:t],'--',color='orange',label='Reference Trajectory')
            ax.plot(self.__xs[start:t:scale],self.__ys[start:t:scale],'o-',color=(0.878, 0.867, 0.137,0.5),markerfacecolor=(0.878, 0.867, 0.137,1.0),markeredgecolor=(0.878, 0.867, 0.137,1.0),linewidth=8,markersize=2,label='Active Window')
            if k>=0:
                ellipses = Ellipses[k]
                ax.plot(predicted_X_pos[k,:],predicted_Y_pos[k,:],'go-',label='Predicted Trajectory',markersize=2,linewidth=1)
                ax.plot(
                    [self.__xs[t],predicted_X_pos[k,0]],
                    [self.__ys[t],predicted_Y_pos[k,0]],
                    'go-',markersize=2,linewidth=1
                )
                for ellipse in ellipses:
                    ax.add_patch(ellipse)
            ax.plot(np.NaN, np.NaN,'c-',linewidth=5,alpha=0.5,label='Confidence')
            ax.legend()

        anim = animation.FuncAnimation(fig,animation_function,frames=int(self.__duration/scale),interval=100,repeat=False)

        if show:
            plt.show()
        if save:
            anim.save(f'imgs/animations/sin-wind-{kernel_name}-{self.__trajectory_name}-{horizon}-step-prediction.gif',writer=animation.FFMpegWriter(fps=30))
        plt.close('all')

    def simulate_mogp(self, window_size, predictor, p0=None, show=False, save=None, kernel_name=''):
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        kernel = RBFKernel(
            predictor.covar_module.data_covar_module.base_kernel.lengthscale.item(),
            predictor.covar_module.data_covar_module.outputscale.item(),
        )
        predictor = GPModel(
            kernel,
            predictor.likelihood.noise.item(),
            2,
            3,
            window_size,
        )
        
        self.__idx_control = []

        # Set the mass initial conditions
        target_p, target_v = self.__trajectory.trajectory()
        if p0 is None:
            x0 = target_p[0,0]
            y0 = target_p[1,0]
            z0 = 0.0
        else:
            x0 = p0[0]
            y0 = p0[1]
            z0 = p0[2]
        self.__quadrotor.set_state(
            np.array([
                x0,y0,z0,0,0,0,
                0,0,0,0
            ])
        )

        # Simulate the field 
        k = 0
        control_force = np.zeros((4,1))
        print(f'Simulating {self.__trajectory_name} Trajectory...')
        for t in range(len(self.__trajectory)):
            print(
                '|{}{}| {:.2f}% ({:.2f}/{:.2f} s)'.format(
                    '█'*int(20*(t+1)/len(target_p[0,:])),
                    ' '*(20-int(20*(t+1)/len(target_p[0,:]))),
                    (t+1)/len(target_p[0,:])*100,
                    (t+1)*self.__dt,
                    len(target_p[0,:])*self.__dt
                ),
                end='\r'
            )
            total_speed = np.array([0,0],dtype=float)
            state = self.__quadrotor.get_state()
            for fan in self.fans:
                speed = fan.generate_wind(state[0],state[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = 0.1*total_speed*np.sign(total_speed)
            ep = target_p[:,t] - state[:3]
            ev = target_v[:,t] - state[3:6]
            if t%self.__control_frequency == 0:
                self.__idx_control.append(t)
                # Collect inputs for GP
                state = self.__quadrotor.get_state()
                self.__gp_data.append([state[0],state[1]])
                # Generate MPC Reference
                idx = min(t+self.__control_horizon*self.__control_frequency,len(self.__trajectory))
                ref = np.concatenate([
                        target_p[:,t:idx:self.__control_frequency],
                        target_v[:,t:idx:self.__control_frequency]
                ],axis=0)
                # If the remaining trajectory is < than the control horizon
                # expand it using the last refence
                if (idx-t)//self.__control_frequency < self.__control_horizon:
                    ref = np.concatenate([
                        ref,
                        np.repeat(ref[:,-1,np.newaxis],self.__control_horizon-(idx-t)//self.__control_frequency,axis=1)
                    ],axis=1)
                # Generate control force
                control_force, _ = self.__mpc(state,ref)
                
                self.__ctl_phi.append(control_force[0])
                self.__ctl_theta.append(control_force[1])
                self.__ctl_psi.append(control_force[2])
                self.__ctl_a.append(control_force[3])

                # Collect labels for GP
                self.__gp_label_x.append(wind_force[0])
                self.__gp_label_y.append(wind_force[1])
                p = np.array(state[:2])
                if k == window_size:
                    self.__mpc = MPC(
                        self.__quadrotor.get_dynamics(),
                        self.__control_horizon,
                        self.__dt*self.__control_frequency,
                        Q=100*np.eye(6), 
                        R=0.1*np.eye(4),
                        input_dim=2,
                        output_dim=3,
                        window_size=50,
                        predictor=predictor,
                        obstacles=self.__obstacles
                    )
                    # Update GP Model
                    self.__mpc.update_predictor(
                        p,
                        np.hstack([wind_force,0.0])
                    )
                elif k>window_size:
                    # Update GP Model
                    self.__mpc.update_predictor(
                        p,
                        np.hstack([wind_force,0.0])
                    )
                else:
                    # Update GP Model
                    predictor.update(
                        p,
                        np.hstack([wind_force,0.0])
                    )
                k+=1

            self.__xs.append(state[0])
            self.__ys.append(state[1])
            self.__zs.append(state[2])
            self.__vxs.append(state[3])
            self.__vys.append(state[4])
            self.__vzs.append(state[5])
            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__ez.append(ep[2])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__evz.append(ev[2])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            # Simulate Dynamics
            self.__quadrotor.step(control_force,np.hstack([wind_force,0.0]))

        # Plots
        T = np.linspace(0,self.__duration*self.__dt,self.__duration)
        control_limit_low = self.__mpc.lower
        control_limit_upper = self.__mpc.upper

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        fig.suptitle('System Trajectory')
        ax.plot(x0,y0,z0,'o',color='b',label='Starting Position')
        ax.plot(*target_p,'--',color='g',label='Reference Trajectory')
        ax.plot(self.__xs,self.__ys,self.__zs,color='orange',label="System Trajectory")
        for obstacle in self.__obstacles:
            Xc,Yc,Zc = self.__data_for_cylinder_along_z(
                obstacle.x,
                obstacle.y,
                obstacle.r,
                4
            )
            ax.plot_surface(Xc, Yc, Zc, color='k')
        ax.axis('equal')
        plt.legend()

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=2)
        ax.plot(self.__xs,self.__ys,label='System Trajectory',color='orange')
        ax.plot(x0,y0,'bo',label='Starting Position')
        ax.plot(*target_p[:2],'g--',label='Reference Trajectory')
        ax.plot(
            self.__xs[self.__control_frequency*window_size],
            self.__ys[self.__control_frequency*window_size],
            'ro',
            label='Start of GP Prediction'
        )
        ax.axis('equal')
        for obstacle in self.__obstacles:
            o = Ellipse((obstacle.x,obstacle.y),2*obstacle.r,2*obstacle.r,edgecolor='k',fc='k')
            ax.add_patch(o)
        ax.legend()

        fig, ax = plt.subplots(4,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=4)
        ax[0].plot(T[self.__idx_control],self.__ctl_phi)
        ax[0].axhline(y=control_limit_low[0],color='k',linestyle='dashed',label=r'$\phi^{\lim}$')
        ax[0].axhline(y=control_limit_upper[0],color='k',linestyle='dashed')
        ax[0].set_ylabel(r'$\phi^c$ $[rad]$')
        ax[0].legend()
        ax[0].set_xlim([0,self.__duration*self.__dt])
        ax[1].plot(T[self.__idx_control],self.__ctl_theta)
        ax[1].axhline(y=control_limit_low[1],color='k',linestyle='dashed',label=r'$\theta^{\lim}$')
        ax[1].axhline(y=control_limit_upper[1],color='k',linestyle='dashed')
        ax[1].set_ylabel(r'$\theta^c$ $[rad]$')
        ax[1].legend()
        ax[1].set_xlim([0,self.__duration*self.__dt])
        ax[2].plot(T[self.__idx_control],self.__ctl_psi)
        ax[2].axhline(y=control_limit_low[2],color='k',linestyle='dashed',label=r'$\psi^{\lim}$')
        ax[2].axhline(y=control_limit_upper[2],color='k',linestyle='dashed')
        ax[2].set_ylabel(r'$\psi^c$ $[rad]$')
        ax[2].legend()
        ax[2].set_xlim([0,self.__duration*self.__dt])
        ax[3].plot(T[self.__idx_control],self.__ctl_a)
        ax[3].axhline(y=control_limit_low[3],color='k',linestyle='dashed',label=r'$a^{\lim}$')
        ax[3].axhline(y=control_limit_upper[3],color='k',linestyle='dashed')
        ax[3].set_ylabel(r'$a^c$ $[m/s]$')
        ax[3].set_xlabel(r'$t$ $[s]$')
        ax[3].legend()
        ax[3].set_xlim([0,self.__duration*self.__dt])

        plt.show()

        print('')

    def simulate_goal_position(self, window_size, predictor, p0, xf, show=False, save=None, kernel_name=''):
        
        kernel = RBFKernel(
            predictor.covar_module.data_covar_module.base_kernel.lengthscale.item(),
            predictor.covar_module.data_covar_module.outputscale.item(),
        )
        predictor = GPModel(
            kernel,
            predictor.likelihood.noise.item(),
            2,
            3,
            window_size,
        )

        self.__idx_control = []

        x0 = p0[0]
        y0 = p0[1]
        z0 = p0[2]

        self.__quadrotor.set_state(
            np.array([
                x0,y0,z0,0,0,0,
                0,0,0,0
            ])
        )

        # Simulate the field 
        k = 0
        control_force = np.zeros((4,1))
        for t in range(self.__duration):
            print(
                '|{}{}| {:.2f}% ({:.2f}/{:.2f} s)'.format(
                    '█'*int(20*(t+1)/self.__duration),
                    ' '*(20-int(20*(t+1)/self.__duration)),
                    (t+1)/self.__duration*100,
                    (t+1)*self.__dt,
                    self.__duration*self.__dt
                ),
                end='\r'
            )
            total_speed = np.array([0,0],dtype=float)
            state = self.__quadrotor.get_state()
            for fan in self.fans:
                speed = fan.generate_wind(state[0],state[1],t*self.__dt)
                total_speed+=speed

            # Generate wind force
            wind_force = 0.1*total_speed*np.sign(total_speed)
            ep = xf[:3] - state[:3]
            ev = xf[3:] - state[3:6]
            if t%self.__control_frequency == 0:
                self.__idx_control.append(t)
                # Collect inputs for GP
                state = self.__quadrotor.get_state()
                self.__gp_data.append([state[0],state[1]])
                # Generate MPC Reference
                ref = np.repeat(xf[:,np.newaxis],self.__control_horizon,1)
                # Generate control force
                control_force, _ = self.__mpc(state,ref)
                
                self.__ctl_phi.append(control_force[0])
                self.__ctl_theta.append(control_force[1])
                self.__ctl_psi.append(control_force[2])
                self.__ctl_a.append(control_force[3])

                # Collect labels for GP
                self.__gp_label_x.append(wind_force[0])
                self.__gp_label_y.append(wind_force[1])
                p = np.array(state[:2])
                if k == window_size:
                    self.__mpc = MPC(
                        self.__quadrotor.get_dynamics(),
                        self.__control_horizon,
                        self.__dt*self.__control_frequency,
                        Q=100*np.eye(6), 
                        R=0.1*np.eye(4),
                        input_dim=2,
                        output_dim=3,
                        window_size=50,
                        predictor=predictor,
                        obstacles=self.__obstacles
                    )
                    # Update GP Model
                    self.__mpc.update_predictor(
                        p,
                        np.hstack([wind_force,0.0])
                    )
                elif k>window_size:
                    # Update GP Model
                    self.__mpc.update_predictor(
                        p,
                        np.hstack([wind_force,0.0])
                    )
                else:
                    # Update GP Model
                    predictor.update(
                        p,
                        np.hstack([wind_force,0.0])
                    )
                k+=1

            self.__xs.append(state[0])
            self.__ys.append(state[1])
            self.__zs.append(state[2])
            self.__vxs.append(state[3])
            self.__vys.append(state[4])
            self.__vzs.append(state[5])
            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__ez.append(ep[2])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__evz.append(ev[2])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            # Simulate Dynamics
            self.__quadrotor.step(control_force,np.hstack([wind_force,0.0]))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=5)
        fig.suptitle('System Trajectory')
        ax.plot(*p0,'o',color='b',label='Starting Position')
        ax.plot(*xf[:3],'o',color='g',label='Target Position')
        ax.plot(self.__xs,self.__ys,self.__zs,color='orange',label="System Trajectory")
        for obstacle in self.__obstacles:
            Xc,Yc,Zc = self.__data_for_cylinder_along_z(
                obstacle.x,
                obstacle.y,
                obstacle.r,
                4
            )
            ax.plot_surface(Xc, Yc, Zc, color='k')
        plt.legend()

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=2)
        ax.plot(self.__xs,self.__ys,label='System Trajectory',color='orange')
        ax.plot(*p0[:2],'bo',label='Starting Position')
        ax.plot(*xf[:2],'go',label='Target Position')
        ax.plot(
            self.__xs[self.__control_frequency*window_size],
            self.__ys[self.__control_frequency*window_size],
            'ro',
            label='Start of GP Prediction'
        )
        ax.axis('equal')
        for obstacle in self.__obstacles:
            o = Ellipse((obstacle.x,obstacle.y),obstacle.r,obstacle.r,edgecolor='k',fc='k')
            ax.add_patch(o)
        ax.legend()

        fig, ax = plt.subplots(3,1)
        fig.tight_layout(pad=2)
        fig.set_size_inches(16,9)
        T = np.linspace(0,self.__duration*self.__dt,self.__duration)
        ax[0].plot(T,self.__xs,color='orange')
        ax[0].axhline(y=xf[0])
        ax[1].plot(T,self.__ys,color='orange')
        ax[1].axhline(y=xf[1])
        ax[2].plot(T,self.__zs,color='orange')
        ax[2].axhline(y=xf[2])

        plt.show()

        print('')

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
        
        self.__setup_system()

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

    def __data_for_cylinder_along_z(self,center_x,center_y,radius,height_z):
        z = np.linspace(0, height_z, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        return x_grid,y_grid,z_grid