import numpy as np
import matplotlib.pyplot as plt
import json
import torch

from matplotlib import animation
from pathlib import Path

from modules.Fan import Fan
from modules.System import System
from modules.Trajectory import Trajectory
from modules.PD import PD
from utils.exceptions import MissingTrajectory

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

        ## Parse fans' data
        self.__fans = []
        for fan in data["fans"]:
            x0 = float(fan["x0"])
            y0 = float(fan["y0"])
            alpha = np.deg2rad(float(fan["alpha"]))
            theta = float(fan["theta"])
            v0 = float(fan["v0"])
            noise_var = float(fan['noise_var'])
            length = float(fan["length"])

            u0 = np.array([1,0])
            rot_mat = np.array([
                [np.cos(alpha),-np.sin(alpha)],
                [np.sin(alpha),np.cos(alpha)]
            ],dtype=float)
            u0 = rot_mat@u0

            # Move the fan to increase the spread in the origin
            h = 0.5*length/np.tan(theta/2)
            x0 = x0-h*u0[0]
            y0 = y0-h*u0[1]

            f = Fan(x0,y0,u0[0],u0[1],theta,v0,noise_var)
            self.__fans.append(f)
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
        Kp = np.diag([506.0,420.0])
        Kd = np.diag([45.0,41.0])
        # The controller's parameter were retrieved using MATLAB
        self.__pd = PD(Kp,Kd)
        
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

    def __draw_wind_field_grid(self):
        vxs = []
        vys = []
        vs = []
        for x in np.linspace(0.1,self.__width-0.1,self.__grid_resolution):
            vx = []
            vy = []
            v = []
            for y in np.linspace(0.1,self.__height-0.1,self.__grid_resolution):
                total_speed = np.zeros((2,),dtype=float)
                for fan in self.__fans:
                    total_speed+=fan.generate_wind(x,y)
                vx.append(total_speed[0])
                vy.append(total_speed[1])
                v.append(np.linalg.norm(total_speed))
            vxs.append(vx)
            vys.append(vy)
            vs.append(v)
        return np.linspace(0.1,self.__width-0.1,self.__grid_resolution),np.linspace(0.1,self.__height-0.1,self.__grid_resolution),np.array(vxs),np.array(vys), np.array(vs)

    def set_trajectory(self, trajectory_file,trajectory_name):
        # Generate Trajectory
        self.__trajectory = Trajectory(trajectory_file)
        self.__trajectory_name = trajectory_name
        self.__tr_p, self.__tr_v = self.__trajectory.trajectory()

    def simulate_wind_field(self): 
        '''
        Runs the wind simulation. The wind field should be reset every time a new simulation.
        In case a GP model is being trained, the GP data should not be reset, as it stacks the subsequent
        measurements which can be used for training.
        '''
        if self.__trajectory is None:
            raise MissingTrajectory()

        # Set the mass initial conditions
        p,v = self.__trajectory.trajectory()
        x0 = p[0,0]
        y0 = p[1,0]
        self.__system.p[0] = x0
        self.__system.p[1] = y0

        # Simulate the field 
        for target_p, target_v in self.__trajectory:
            total_speed = np.array([0,0],dtype=float)
            for fan in self.__fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1])
                total_speed+=speed

            # Collect inputs for GP
            self.__gp_data.append([self.__system.p[0],self.__system.p[1]])
            
            # Generate control force
            ep = target_p - self.__system.p
            ev = target_v - self.__system.v
            control_force = self.__pd.step(ep,ev)
            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            # Total force
            force = wind_force + control_force
            # Add GP prediction wind force
            if self.__gp_predictor_x is not None and self.__gp_predictor_y is not None:
                with torch.no_grad():
                    p = torch.FloatTensor([[self.__system.p[0],self.__system.p[1]]])
                    predicted_wind_force_x = self.__gp_predictor_x(p).mean.item()
                    predicted_wind_force_y = self.__gp_predictor_y(p).mean.item()
                force -= np.array([predicted_wind_force_x,predicted_wind_force_y],dtype=float)
            
            self.__xs.append(self.__system.p[0])
            self.__ys.append(self.__system.p[1])
            self.__vxs.append(self.__system.v[0])
            self.__vys.append(self.__system.v[1])
            self.__ctl_forces_x.append(control_force[0])
            self.__ctl_forces_y.append(control_force[1])
            self.__ex.append(ep[0])
            self.__ey.append(ep[1])
            self.__evx.append(ev[0])
            self.__evy.append(ev[1])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            # Collect labels for GP
            self.__gp_label_x.append(wind_force[0])
            self.__gp_label_y.append(wind_force[1])

            # Simulate Dynamics
            self.__system.discrete_dynamics(force)

    def reset(self, wind_field_conf_file=None, mass_conf_file=None):
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

        self.__setup_plots()

    def get_gp_data(self):
        '''
        Returns the GP data needed for training or testing.\\
        The data is in the form (x,y), Fx, Fy, T
        '''
        return self.__gp_data.copy(), self.__gp_label_x.copy(), self.__gp_label_y.copy(), [t*self.__dt for t in range(self.__duration)]
    
    def get_wind_field_data(self):
        
        x, y, vx, vy, _ = self.__draw_wind_field_grid()
        fx = (0.5*self.__air_density*self.__system.surf)*vx**2*np.sign(vx)
        fy = (0.5*self.__air_density*self.__system.surf)*vy**2*np.sign(vy)
        f = np.sqrt(np.power(fx+fy,2))

        return {
            'x': x,
            'y': y,
            'fx': fx,
            'fy': fy,
            'f': f
        }
    
    def reset_gp(self):
        '''
        Resets the GP data
        '''
        self.__setup_gp()

    def plot(self, save=False, folder='imgs/trajectories_plots'):
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
        if save:
            plt.savefig(folder+f'/{file_name}-trajectory-x-position.png',dpi=300)
            plt.savefig(folder+f'/{file_name}-trajectory-x-position.svg')

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
        if save:
            plt.savefig(folder+f'/{file_name}-trajectory-y-position.png',dpi=300)
            plt.savefig(folder+f'/{file_name}-trajectory-y-position.svg')

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
        if save:
            plt.savefig(folder+f'/{file_name}-trajectory-traking.png',dpi=300)
            plt.savefig(folder+f'/{file_name}-trajectory-traking.svg')

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
        if save:
            plt.savefig(folder+f'/{file_name}-trajectory-velocity.png',dpi=300)
            plt.savefig(folder+f'/{file_name}-trajectory-velocity.svg')

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__ctl_forces_x)
        ax[0].set_xlabel(r'$t$ $[s]$')
        ax[0].set_ylabel(r'$u_x$ $[N]$')
        ax[0].title.set_text(r'Control Force ($u_x$)')
        ax[1].plot(T,self.__ctl_forces_y)
        ax[1].set_xlabel(r'$t$ $[s]$')
        ax[1].set_ylabel(r'$u_y$ $[N]$')
        ax[1].title.set_text(r'Control Force ($u_y$)')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save:
            plt.savefig(folder+f'/{file_name}-trajectory-control-force.png',dpi=300)
            plt.savefig(folder+f'/{file_name}-trajectory-control-force.svg')

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
        if save:
            plt.savefig(folder+f'/{file_name}-trajectory-wind-force.png',dpi=300)
            plt.savefig(folder+f'/{file_name}-trajectory-wind-force.svg')

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        xs, ys, vx, vy, v = self.__draw_wind_field_grid()
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        for i in range(len(xs)):
            for j in range(len(ys)):
                if v[i,j]>0.2:
                    ax.arrow(xs[i],ys[j],vx[i,j]/40,vy[i,j]/40,length_includes_head=False,head_width=0.015,head_length=0.015,width=0.003,color='orange')
                else:
                    ax.plot(xs[i],ys[j],'o',color='orange',markersize=5*v[i,j])
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        ax.set_title('Wind Field')
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        fig.legend(['Wind Speed'])
        if save:
            plt.savefig(folder+f'/wind-field.png',dpi=300)
            plt.savefig(folder+f'/wind-field.svg')

        plt.show()
        plt.close()

    def animate(self):
        '''
        Plots the animation showing the evolution of the system following the trajectory
        in the wind field
        '''
        if not self.__xs:
            print('No data to plot!')
            return
        
        file_name = self.__trajectory_name

        fig, ax = plt.subplots()
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        def animation_function(t):
            ax.clear()
            ax.set_xlim([0,self.__width])
            ax.set_ylim([0,self.__height])
            ax.plot(np.NaN, np.NaN, '-', color='none', label='t={0:.2f} s'.format(t*self.__dt))
            ax.plot(self.__tr_p[0,t],self.__tr_p[1,t],'o',color='orange',markersize=7,label='Target Distance=[{0:.2f},{0:.2f}] m'.format(self.__ex[t],self.__ey[t])) # Traget Location
            ax.plot(self.__xs[t],self.__ys[t],'bo',markersize=5) # Object Moving
            ax.quiver(self.__xs[t],self.__ys[t],self.__wind_force_x[t],self.__wind_force_y[t],scale=20,width=0.003,color='r',label='Wind Force=[{0:.2f},{0:.2f}] N'.format(self.__wind_force_x[t],self.__wind_force_y[t])) # Wind Force
            ax.quiver(self.__xs[t],self.__ys[t],self.__ctl_forces_x[t],self.__ctl_forces_y[t],scale=20,width=0.003,color="#4DBEEE",label='Control Force=[{0:.2f},{0:.2f}] N'.format(self.__ctl_forces_x[t],self.__ctl_forces_y[t])) # Control Force
            ax.plot(self.__xs[:t],self.__ys[:t],'b')
            ax.plot(self.__tr_p[0,:t],self.__tr_p[1,:t],'--',color='orange')
            # Plot fans
            for fan in self.__fans:
                ax.quiver(fan.p0[0],fan.p0[1],fan.u0[0],fan.u0[1],scale=10,color='k')
            ax.legend()

        anim = animation.FuncAnimation(fig,animation_function,frames=self.__duration,interval=1,repeat=False)

        plt.show()
        plt.close()