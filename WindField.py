import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import sys
import os
import casadi as ca

from scipy.stats import chi2

from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from modules.Fan import RealFan, SimulatedFan
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
        if data["fans"]["type"] == "simulated":
            for fan in data["fans"]["src"]:
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
                f = SimulatedFan(x0,y0,u0[0],u0[1],length,noise_var,generator_function,self.__obstacles)
                self.fans.append(f)
        elif data["fans"]["type"] == "real":
            src_mean = data["fans"]["src"]["mean"]
            src_var = data["fans"]["src"]["var"]
            scale_factor = data["fans"]["src"]["scale_factor"]
            mean_map = np.load(src_mean)
            var_map = np.load(src_var)
            f = RealFan(mean_map,var_map,self.__width,self.__height,mean_map.shape[1],scale_factor)
            self.fans.append(f)

        file.close()

    def __setup_system(self):
        self.__quadrotor = Quadrotor(
            self.__dt,
            np.zeros(10)
        )
        self.__control_horizon = 10      
        
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
        self.__phi = []
        self.__theta = []
        self.__psi = []
        self.__a = []

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
        if show:
            plt.show()
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-traking.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-traking.svg')

        fig, ax = plt.subplots(3,1)
        fig.tight_layout(pad=2)
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
        fig.tight_layout(pad=2)
        ax[0].plot(T,self.__ctl_phi)
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$\phi^c$ $[red]$')
        ax[0].title.set_text(r'Control $\phi$ ($\phi^c$)')
        ax[1].plot(T,self.__ctl_theta)
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$\theta^c$ $[red]$')
        ax[1].title.set_text(r'Control $\theta$ ($\theta^c$)')
        ax[2].plot(T,self.__ctl_psi)
        ax[2].set_xlabel(r'$t$ $[s]$')
        ax[2].set_ylabel(r'$\psi^c$ $[red]$')
        ax[2].title.set_text(r'Control $\psi$ ($\psi^c$)')
        ax[3].plot(T,self.__ctl_a)
        ax[3].set_xlabel(r'$t$ $[s]$')
        ax[3].set_ylabel(r'$a_c$ $[m/s^2]$')
        ax[3].title.set_text(r'Control Thrust ($a^c$)')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save is not None:
            plt.savefig(save+f'/{file_name}-trajectory-control-action.png',dpi=300)
            plt.savefig(save+f'/{file_name}-trajectory-control-action.svg')

        fig, ax = plt.subplots(1,2)
        fig.tight_layout(pad=2)
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

    def simulate_wind_field(self): 
        '''
        Runs the wind simulation. The wind field should be reset every time a new simulation.
        In case a GP model is being trained, the GP data should not be reset, as it stacks the subsequent
        measurements which can be used for training.
        '''
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        self.__mpc = MPC(
            self.__quadrotor,
            self.__control_horizon,
            self.__dt*self.__control_frequency,
            Q=100*np.eye(6), 
            R=np.eye(4),
            maximum_solver_time=self.__dt*self.__control_frequency,
            obstacles=self.__obstacles
        )  

        # Set the mass initial conditions
        target_p,target_v = self.__trajectory.trajectory()
        self.__quadrotor.set_state(np.array([
            target_p[0,0], target_p[1,0], target_p[2,0],
            0,0,0,
            0,0,0,9.81
        ]))
        prev_x_opt = np.vstack([
            target_p[:,:self.__control_horizon],
            target_v[:,:self.__control_horizon],
            np.zeros((3,self.__control_horizon)),
            9.81*np.ones((1,self.__control_horizon))
        ])
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
            wind_force = 0.5*0.47*1.225*np.pi*self.__quadrotor.r**2*total_speed**2*np.sign(total_speed)
            ep = target_p[:,t] - state[:3]
            ev = target_v[:,t] - state[3:6]
            if t%self.__control_frequency == 0:
                self.__idx_control.append(t)
                # Collect inputs for GP
                state = self.__quadrotor.get_state()
                self.__gp_data.append([state[0],state[1]])
                # Generate MPC Reference
                idx = min(t+(self.__control_horizon+1)*self.__control_frequency,len(self.__trajectory))
                ref = np.concatenate([
                        target_p[:,t:idx:self.__control_frequency],
                        target_v[:,t:idx:self.__control_frequency]
                ],axis=0)
                # If the remaining trajectory is < than the control horizon
                # expand it using the last refence
                if (idx-t)//self.__control_frequency < (self.__control_horizon+1):
                    ref = np.concatenate([
                        ref,
                        np.repeat(ref[:,-1,np.newaxis],(self.__control_horizon+1)-(idx-t)//self.__control_frequency,axis=1)
                    ],axis=1)
                # Generate control force
                control_force, x_opt, _ = self.__mpc(state,ref,prev_x_opt)
                prev_x_opt = x_opt[:,1:]

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
            self.__ctl_phi.append(control_force[0])
            self.__ctl_theta.append(control_force[1])
            self.__ctl_psi.append(control_force[2])
            self.__ctl_a.append(control_force[3])

            # Simulate Dynamics
            self.__quadrotor.step(control_force,np.hstack([wind_force,0]))

        # Animation
        ## Add Obstacles
        obstacles = []
        for obstacle in self.__obstacles:
            o = Ellipse((obstacle.x,obstacle.y),2*obstacle.r,2*obstacle.r,edgecolor='k',fc='k')
            obstacles.append(o)

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=2)

        DroneEllipses = []
        for i in range(int(self.__duration)):
            DroneEllipses.append(
                Ellipse(
                    (self.__xs[i],self.__ys[i]),
                    2*self.__quadrotor.r,
                    2*self.__quadrotor.r,
                    fc='firebrick',
                    edgecolor='firebrick',
                    alpha=0.4
                )
            )

        render_full_animation = True
        scale = 2

        if render_full_animation:
            _, _, _, _, v = self.__draw_wind_field_grid()
            v_max = np.max(v)
            # Create custom colormap
            colors = [(1, 0.5, 0, alpha) for alpha in np.linspace(0, 1, 256)]
            orange_transparency_cmap = LinearSegmentedColormap.from_list('orange_transparency', colors, N=256)
            bar = ax.imshow(np.array([[0,v_max]]), cmap=orange_transparency_cmap)
            bar.set_visible(False)
            cb = fig.colorbar(bar,orientation="vertical")
            cb.set_label(label=r'Wind Speed $[m/s]$',labelpad=10)
            ax.set_xlim([0.0,self.__width])
            ax.set_ylim([0.0,self.__height])

        def animation_function(t):
            # Clear figures and setup plots
            ax.clear()

            t = int(t*self.__control_frequency*scale)
            ax.set_xlim([0.0,self.__width])
            ax.set_ylim([0.0,self.__height])
            ax.set_aspect('equal','box')
            if render_full_animation:
                xs, ys, vx, vy, v = self.__draw_wind_field_grid()
                v_max = np.max(v)
                for i in range(len(xs)):
                    for j in range(len(ys)):
                        ax.arrow(xs[i],ys[j],vx[i,j]/(v_max*10),vy[i,j]/(v_max*10),length_includes_head=False,head_width=0.015,head_length=0.015,width=0.005,color='orange',alpha=v[i,j]/v_max)

            # Plot System Evolution
            for o in obstacles:
                ax.add_patch(o)
            ax.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*self.__dt))
            ax.plot(np.NaN, np.NaN, 'o', color='k', markersize=10, label='Obstacles')
            ax.plot(np.NaN, np.NaN, '-', color='firebrick', alpha=0.4, linewidth=8, label='Drone Radius')
            ax.plot(target_p[0,:t],target_p[1,:t],'--',color='orchid',label="Reference Trajectory")
            ax.plot(target_p[0,t],target_p[1,t],'o',color='orchid')
            ax.plot(self.__xs[t],self.__ys[t],'o',color='tab:blue',label='System Position')
            ax.plot(self.__xs[:t],self.__ys[:t],color="tab:blue",label="System Trajectory",alpha=0.8)
            ax.add_patch(DroneEllipses[t])

            ax.legend()
            
        anim = animation.FuncAnimation(fig,animation_function,frames=int(self.__duration/(self.__control_frequency*scale)),interval=10,repeat=False)
        FFwriter = animation.FFMpegWriter(fps=30)
        if render_full_animation:
            print(f"Rendering {self.__trajectory_name} Trajectory animation")
            anim.save(f'imgs/animations/{self.__trajectory_name}_no_gp_{len(self.__obstacles)}.obs.mp4', writer = FFwriter)
            print("Done rendering!")
        else:
            plt.show()
        plt.close('all')
        
        print('')

    def simulate_mogp(self, window_size, predictor, p0=None, show=False, save=None, kernel_name=''):
        if self.__trajectory is None:
            raise MissingTrajectoryException()
        
        kernel = RBFKernel(
            predictor.covar_module.data_covar_module.base_kernel.lengthscale.item(),
            predictor.covar_module.data_covar_module.outputscale.item(),
        )

        predictor = GPModel(
            kernel,
            # predictor.likelihood.noise.item(),
            0.01,
            2,
            3,
            window_size,
        )

        self.__mpc = MPC(
            self.__quadrotor,
            self.__control_horizon,
            self.__dt*self.__control_frequency,
            Q=np.diag([10,10,10,2,2,2]), 
            R=np.diag([1,1,1,1]),
            maximum_solver_time=self.__dt*self.__control_frequency,
            obstacles=self.__obstacles,
            predictor=predictor
        )     
        
        # Setup Plots 
        self.__idx_control = []
        Covs = []
        PredictedPos = []

        # Set the initial conditions
        target_p, target_v = self.__trajectory.trajectory()
        if p0 is None:
            x0 = target_p[0,0]
            y0 = target_p[1,0]
            z0 = target_p[2,0]
        else:
            x0 = p0[0]
            y0 = p0[1]
            z0 = p0[2]

        # Setup initial guess for the solver
        prev_x_opt = np.vstack([
            target_p[:,:self.__control_horizon],
            target_v[:,:self.__control_horizon],
            np.zeros((3,self.__control_horizon)),
            9.81*np.ones((1,self.__control_horizon))
        ])
        
        self.__quadrotor.set_state(
            np.array([
                x0,y0,z0,0,0,0,
                0,0,0,9.81
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
            wind_force = 0.5*0.47*1.225*np.pi*self.__quadrotor.r**2*total_speed**2*np.sign(total_speed)
            ep = target_p[:,t] - state[:3]
            ev = target_v[:,t] - state[3:6]
            if t%self.__control_frequency == 0:
                self.__idx_control.append(t)
                state = self.__quadrotor.get_state()
                # Generate MPC Reference
                idx = min(t+(self.__control_horizon+1)*self.__control_frequency,len(self.__trajectory))
                ref = np.concatenate([
                        target_p[:,t:idx:self.__control_frequency],
                        target_v[:,t:idx:self.__control_frequency]
                ],axis=0)
                # If the remaining trajectory is < than the control horizon
                # expand it using the last refence
                if (idx-t)//self.__control_frequency < (self.__control_horizon+1):
                    ref = np.concatenate([
                        ref,
                        np.repeat(ref[:,-1,np.newaxis],(self.__control_horizon+1)-(idx-t)//self.__control_frequency,axis=1)
                    ],axis=1)
                # Generate control force
                control_force, predicted_state, pos_cov = self.__mpc(state,ref,prev_x_opt)
                prev_x_opt = predicted_state[:,1:]

                # If the gp prediction is already on, add the covariance on the position for the plots
                if pos_cov is not None:
                    Covs.append(pos_cov.copy())
                    PredictedPos.append(predicted_state[:2,1:].copy())

                p = np.array(state[:2])
                # If the gp model is full, update set the solver to predict the wind
                if k == window_size:
                    self.__mpc.set_predictor()
                # Update GP Model
                self.__mpc.update_predictor(
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
            self.__phi.append(state[6])
            self.__theta.append(state[7])
            self.__psi.append(state[8])
            self.__a.append(state[9])
            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__ez.append(ep[2])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__evz.append(ev[2])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])
            self.__ctl_phi.append(control_force[0])
            self.__ctl_theta.append(control_force[1])
            self.__ctl_psi.append(control_force[2])
            self.__ctl_a.append(control_force[3])

            # Simulate Dynamics
            self.__quadrotor.step(control_force,np.hstack([wind_force,0.0]))

        print('')

        # Plots
        T = np.linspace(0,self.__duration*self.__dt,self.__duration)
        control_limit_low = self.__mpc.lower
        control_limit_upper = self.__mpc.upper

        ## Plot 3D Trajectory
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
        ax.set_aspect('equal','box')
        plt.legend()

        ## Plot 2D xy Projection of the Trajectory
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
        ax.set_aspect('equal','box')
        for obstacle in self.__obstacles:
            o = Ellipse((obstacle.x,obstacle.y),2*obstacle.r,2*obstacle.r,edgecolor='k',fc='k')
            ax.add_patch(o)
        ax.legend()

        ## Plot Control Inputs
        fig, ax = plt.subplots(4,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=4)
        fig.suptitle('Control Inputs')
        ax[0].plot(T,self.__ctl_phi)
        ax[0].axhline(y=control_limit_low[0],color='k',linestyle='dashed',label=r'$\phi^{\lim}$')
        ax[0].axhline(y=control_limit_upper[0],color='k',linestyle='dashed')
        ax[0].set_ylabel(r'$\phi^c$ $[rad]$')
        ax[0].legend()
        ax[0].set_xlim([0,self.__duration*self.__dt])
        ax[1].plot(T,self.__ctl_theta)
        ax[1].axhline(y=control_limit_low[1],color='k',linestyle='dashed',label=r'$\theta^{\lim}$')
        ax[1].axhline(y=control_limit_upper[1],color='k',linestyle='dashed')
        ax[1].set_ylabel(r'$\theta^c$ $[rad]$')
        ax[1].legend()
        ax[1].set_xlim([0,self.__duration*self.__dt])
        ax[2].plot(T,self.__ctl_psi)
        ax[2].axhline(y=control_limit_low[2],color='k',linestyle='dashed',label=r'$\psi^{\lim}$')
        ax[2].axhline(y=control_limit_upper[2],color='k',linestyle='dashed')
        ax[2].set_ylabel(r'$\psi^c$ $[rad]$')
        ax[2].legend()
        ax[2].set_xlim([0,self.__duration*self.__dt])
        ax[3].plot(T,self.__ctl_a)
        ax[3].axhline(y=control_limit_low[3],color='k',linestyle='dashed',label=r'$a^{\lim}$')
        ax[3].axhline(y=control_limit_upper[3],color='k',linestyle='dashed')
        ax[3].set_ylabel(r'$a^c$ $[m/s^2]$')
        ax[3].set_xlabel(r'$t$ $[s]$')
        ax[3].legend()
        ax[3].set_xlim([0,self.__duration*self.__dt])

        ## Plot Tracking Error (x-Position)
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=3)
        fig.suptitle('x Position')
        ax[0].plot(T,self.__xs,label='System x Position')
        ax[0].plot(T,target_p[0,:],'--',color='orange',label='Reference x Position')
        ax[1].plot(T,self.__ex,label='Position Error')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$x$ $[m]$')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_x$ $[m]$')
        ax[0].legend()
        ax[1].legend()
        
        ## Plot Tracking Error (y-Position)
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=3)
        fig.suptitle('y Position')
        ax[0].plot(T,self.__ys,color='tab:blue',label='System y Position')
        ax[0].plot(T,target_p[1,:],'--',color='orange',label='Reference y Position')
        ax[1].plot(T,self.__ey,label='Position Error')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$y$ $[m]$')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_y$ $[m]$')
        ax[0].legend()
        ax[1].legend()

        ## Plot Tracking Error (z-Position)
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=3)
        fig.suptitle('z Position')
        ax[0].plot(T,self.__zs,color='tab:blue',label='System z Position')
        ax[0].plot(T,target_p[2,:],'--',color='orange',label='Reference z Position')
        ax[1].plot(T,self.__ez,label='Position Error')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$z$ $[m]$')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_z$ $[m]$')
        ax[0].legend()
        ax[1].legend()

        ## Plot Tracking Error (x-Velocity)
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=3)
        fig.suptitle('x Velocity')
        ax[0].plot(T,self.__vxs,color='tab:blue',label='System x Velocity')
        ax[0].plot(T,target_v[0,:],'--',color='orange',label='Reference x Velocity')
        ax[1].plot(T,self.__evx,label='Velocity Error')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$V_x$ $[m/s]$')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{V_x}$ $[m/s]$')
        ax[0].legend()
        ax[1].legend()

        ## Plot Tracking Error (y-Velocity)
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=3)
        fig.suptitle('y Velocity')
        ax[0].plot(T,self.__vys,color='tab:blue',label='System y Velocity')
        ax[0].plot(T,target_v[1,:],'--',color='orange',label='Reference y Velocity')
        ax[1].plot(T,self.__evy,label='Velocity Error')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$V_y$ $[m/s]$')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{V_y}$ $[m/s]$')
        ax[0].legend()
        ax[1].legend()

        ## Plot Tracking Error (z-Velocity)
        fig, ax = plt.subplots(2,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=3)
        fig.suptitle('z Velocity')
        ax[0].plot(T,self.__vzs,color='tab:blue',label='System z Velocity')
        ax[0].plot(T,target_v[2,:],'--',color='orange',label='Reference z Velocity')
        ax[1].plot(T,self.__evz,label='Velocity Error')
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$V_z$ $[m/s]$')
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$e_{V_z}$ $[m/s]$')
        ax[0].legend()
        ax[1].legend()

        ## Plot Attitude State Evolution
        fig, ax = plt.subplots(4,1)
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=4)
        fig.suptitle('Attitude and mass-normalized acceleration evolution')
        ax[0].plot(T,self.__phi)
        ax[0].set_ylabel(r'$\phi$ $[rad]$')
        ax[0].set_xlim([0,self.__duration*self.__dt])
        ax[1].plot(T,self.__theta)
        ax[1].set_ylabel(r'$\theta$ $[rad]$')
        ax[1].set_xlim([0,self.__duration*self.__dt])
        ax[2].plot(T,self.__psi)
        ax[2].set_ylabel(r'$\psi$ $[rad]$')
        ax[2].set_xlim([0,self.__duration*self.__dt])
        ax[3].plot(T,self.__a)
        ax[3].set_ylabel(r'$a$ $[m/s^2]$')
        ax[3].set_xlabel(r'$t$ $[s]$')
        ax[3].set_xlim([0,self.__duration*self.__dt])

        if show:
            plt.show()
        plt.close('all')

        # Animation
        ## Add Obstacles
        obstacles = []
        for obstacle in self.__obstacles:
            o = Ellipse((obstacle.x,obstacle.y),2*obstacle.r,2*obstacle.r,edgecolor='k',fc='k')
            obstacles.append(o)

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        fig.tight_layout(pad=2)

        UncEllipses = []
        DroneEllipses = []
        chi2_val = np.sqrt(chi2.ppf(1-self.__mpc.delta, 2))
        for i in range(len(Covs)):
            unc = []
            drones = []
            cov = Covs[i]
            pos = PredictedPos[i]
            for j in range(self.__mpc.N):
                unc.append(
                    Ellipse(
                        (pos[0,j],pos[1,j]),
                        2*(np.sqrt(cov[j,0])*chi2_val+self.__quadrotor.r),
                        2*(np.sqrt(cov[j+1,1])*chi2_val+self.__quadrotor.r),
                        fc='cyan',
                        edgecolor='cyan',
                        alpha=0.5
                    )
                )
                drones.append(
                    Ellipse(
                        (pos[0,j],pos[1,j]),
                        2*self.__quadrotor.r,
                        2*self.__quadrotor.r,
                        fc='firebrick',
                        edgecolor='firebrick',
                        alpha=0.4
                    )
                )
            UncEllipses.append(unc)
            DroneEllipses.append(drones)

        render_full_animation = False
        scale = 2

        if render_full_animation:
            _, _, _, _, v = self.__draw_wind_field_grid()
            v_max = np.max(v)
            # Create custom colormap
            colors = [(1, 0.5, 0, alpha) for alpha in np.linspace(0, 1, 256)]
            orange_transparency_cmap = LinearSegmentedColormap.from_list('orange_transparency', colors, N=256)
            bar = ax.imshow(np.array([[0,v_max]]), cmap=orange_transparency_cmap)
            bar.set_visible(False)
            cb = fig.colorbar(bar,orientation="vertical")
            cb.set_label(label=r'Wind Speed $[m/s]$',labelpad=10)
            ax.set_xlim([0.0,self.__width])
            ax.set_ylim([0.0,self.__height])

        def animation_function(t):
            # Clear figures and setup plots
            ax.clear()

            t = int(t*self.__control_frequency*scale)
            k = t//self.__control_frequency - window_size - 1
            start = max(0,k-window_size)*self.__control_frequency
            ax.set_xlim([0.0,self.__width])
            ax.set_ylim([0.0,self.__height])
            ax.set_aspect('equal','box')
            if render_full_animation:
                xs, ys, vx, vy, v = self.__draw_wind_field_grid()
                v_max = np.max(v)
                for i in range(len(xs)):
                    for j in range(len(ys)):
                        ax.arrow(xs[i],ys[j],vx[i,j]/(v_max*10),vy[i,j]/(v_max*10),length_includes_head=False,head_width=0.015,head_length=0.015,width=0.005,color='orange',alpha=v[i,j]/v_max)

            # Plot System Evolution
            for o in obstacles:
                ax.add_patch(o)
            ax.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*self.__dt))
            ax.plot(np.NaN, np.NaN, 'o', color='k', markersize=10, label='Obstacles')
            ax.plot(np.NaN, np.NaN, '-', color='cyan', alpha=0.5, linewidth=8, label='Uncertainty')
            ax.plot(np.NaN, np.NaN, '-', color='firebrick', alpha=0.4, linewidth=8, label='Drone Radius')
            ax.plot(target_p[0,:t],target_p[1,:t],'--',color='orchid',label="Reference Trajectory")
            ax.plot(target_p[0,t],target_p[1,t],'o',color='orchid')
            ax.plot(self.__xs[t],self.__ys[t],'o',color='tab:blue',label='System Position')
            ax.plot(self.__xs[start:t:self.__control_frequency],self.__ys[start:t:self.__control_frequency],'o-',color=(0.878, 0.867, 0.137,0.5),markerfacecolor=(0.878, 0.867, 0.137,1.0),markeredgecolor=(0.878, 0.867, 0.137,1.0),linewidth=8,markersize=2,label='Active Window')
            ax.plot(self.__xs[:t],self.__ys[:t],color="tab:blue",label="System Trajectory",alpha=0.8)
            if k>0:
                unc = UncEllipses[k]
                drones = DroneEllipses[k]
                ax.plot([self.__xs[t],*PredictedPos[k][0,:]],[self.__ys[t],*PredictedPos[k][1,:]],'-o',color='g',markersize=2,linewidth=1,label="Predicted Position")
                for i in range(self.__mpc.N):
                    ax.add_patch(unc[i])
                for i in range(self.__mpc.N):
                    ax.add_patch(drones[i])

            ax.legend()
            
        anim = animation.FuncAnimation(fig,animation_function,frames=int(self.__duration/(self.__control_frequency*scale)),interval=10,repeat=False)
        FFwriter = animation.FFMpegWriter(fps=30)
        if render_full_animation:
            print(f"Rendering {self.__trajectory_name} Trajectory animation")
            anim.save(f'imgs/animations/{self.__trajectory_name}_gp_{len(self.__obstacles)}.obs.mp4', writer = FFwriter)
            print("Done rendering!")
        else:
            plt.show()
        plt.close('all')

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