import numpy as np
import matplotlib.pyplot as plt
import json

from matplotlib import animation
from pathlib import Path

from modules.Fan import Fan
from modules.System import System
from modules.Trajectory import Trajectory
from modules.PID import PID
from utils.exceptions import MissingTrajectory

class WindField:
    '''
    Class used to model the wind field and simulate the evolution of the system moving in it.\\
    The Wind Field is constructed by passing it the wind field configuration file, and the 
    mass configuration file.
    '''
    def __init__(self, wind_field_conf_file, mass_conf_file):
        self.__wind_field_conf_file = wind_field_conf_file
        self.__mass_config_file = mass_conf_file
        self.__trajectory = None

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

        ## Parse fans' data
        self.__fans = []
        for fan in data["fans"]:
            x0 = float(fan["x0"])
            y0 = float(fan["y0"])
            alpha = np.deg2rad(float(fan["alpha"]))
            theta = float(fan["theta"])
            v0 = float(fan["v0"])
            noise_var = float(fan['noise_var'])

            u0 = np.array([1,0])
            rot_mat = np.array([
                [np.cos(alpha),-np.sin(alpha)],
                [np.sin(alpha),np.cos(alpha)]
            ],dtype=float)
            u0 = rot_mat@u0

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

        # The controller's parameter were retrieved using MATLAB
        self.__pid = PID(
            16.255496588371, # Proportional
            6.40173078542831, # Integral
            9.79714803790873, # Derivative
            self.__dt # Sampling time
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
        self.__vxs = [] # List of x velocities
        self.__vys = [] # List of y velocities
        self.__ctl_forces_x = [] # List of x control forces
        self.__ctl_forces_y = [] # List of y control forces
        self.__wind_force_x = [] # List of x wind forces
        self.__wind_force_y = [] # List of y wind forces
        self.__ex = [] # List of x position traking errors
        self.__ey = [] # List of y position traking errors

    def __draw_wind_field_grid(self):
        vxs = []
        vys = []
        vs = []
        grid_resolution = 50
        for x in np.linspace(0.1,self.__width-0.1,grid_resolution):
            vx = []
            vy = []
            v = []
            for y in np.linspace(0.1,self.__height-0.1,grid_resolution):
                total_speed = 0
                for fan in self.__fans:
                    total_speed+=fan.generate_wind(x,y)
                vx.append(total_speed[0])
                vy.append(total_speed[1])
                v.append(np.linalg.norm(total_speed))
            vxs.append(vx)
            vys.append(vy)
            vs.append(v)
        return np.linspace(0.1,self.__width-0.1,grid_resolution),np.linspace(0.1,self.__height-0.1,grid_resolution),np.array(vxs),np.array(vys), np.array(vs)

    def set_trajectory(self, trajectory_file,trajectory_name):
        # Generate Trajectory
        self.__trajectory = Trajectory(trajectory_file)
        self.__trajectory_name = trajectory_name
        self.__tr = self.__trajectory.trajectory()

    def simulate_wind_field(self): 
        '''
        Runs the wind simulation. The wind field should be reset every time a new simulation.
        In case a GP model is being trained, the GP data should not be reset, as it stacks the subsequent
        measurements which can be used for training.
        '''
        if self.__trajectory is None:
            raise MissingTrajectory()

        # Set the mass initial conditions
        tr = self.__trajectory.trajectory()
        x0 = tr[0,0]
        y0 = tr[1,0]
        self.__system.p[0] = x0
        self.__system.p[1] = y0

        # Simulate the field 
        for target in self.__trajectory:
            total_speed = np.array([0,0],dtype=float)
            for fan in self.__fans:
                speed = fan.generate_wind(self.__system.p[0],self.__system.p[1])
                total_speed+=speed

            # Collect inputs for GP
            self.__gp_data.append([self.__system.p[0],self.__system.p[1]])
            
            # Generate control force
            error = target - self.__system.p
            control_force = self.__pid.step(error)
            # Generate wind force
            wind_force = (0.5*self.__air_density*self.__system.surf)*total_speed**2*np.sign(total_speed)
            # Total force
            force = wind_force + control_force
            # Simulate Dynamics
            self.__system.discrete_dynamics(force)
            self.__xs.append(self.__system.p[0])
            self.__ys.append(self.__system.p[1])
            self.__vxs.append(self.__system.v[0])
            self.__vys.append(self.__system.v[1])
            self.__ctl_forces_x.append(control_force[0])
            self.__ctl_forces_y.append(control_force[0])
            self.__ex.append(error[0])
            self.__ey.append(error[1])
            self.__wind_force_x.append(wind_force[0])
            self.__wind_force_y.append(wind_force[1])

            # Collect labels for GP
            self.__gp_label_x.append(wind_force[0])
            self.__gp_label_y.append(wind_force[1])

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

    def plot(self, save=False):
        '''
        Plots the data related to the previosly run simulation.\\
        If the save parameter is set to `True`, the files are stored in the
        `imgs/trajectories_plots` folder
        '''
        if not self.__xs:
            print('No data to plot!')
            return

        T = [t*self.__dt for t in range(self.__duration)]
        tr = self.__trajectory.trajectory()
        file_name = Path(self.__trajectory_name).stem

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__xs,label='Object Position')
        ax[0].plot(T,tr[0,:],'--',label='Reference Position')
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
            plt.savefig(f'imgs/trajectories_plots/{file_name}-trajectory-x-position.png',dpi=300)

        fig, ax = plt.subplots(1,2)
        ax[0].plot(T,self.__ys,label='Object Position')
        ax[0].plot(T,tr[1,:],'--',label='Reference Position')
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
            plt.savefig(f'imgs/trajectories_plots/{file_name}-trajectory-y-position.png',dpi=300)

        fig, ax = plt.subplots()
        ax.plot(self.__xs,self.__ys,label='System Trajectory')
        ax.plot(tr[0,:],tr[1,:],'--',label='Trajectory to Track')
        ax.title.set_text(r'Trajectory')
        ax.legend()
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        fig.suptitle(f'{file_name} Trajectory')
        fig.set_size_inches(16,9)
        if save:
            plt.savefig(f'imgs/trajectories_plots/{file_name}-trajectory-traking.png',dpi=300)

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
            plt.savefig(f'imgs/trajectories_plots/{file_name}-trajectory-velocity.png',dpi=300)

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
            plt.savefig(f'imgs/trajectories_plots/{file_name}-trajectory-control-force.png',dpi=300)

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
            plt.savefig(f'imgs/trajectories_plots/{file_name}-trajectory-wind-force.png',dpi=300)

        fig, ax = plt.subplots()
        fig.set_size_inches(16,9)
        xs, ys, vx, vy, v = self.__draw_wind_field_grid()
        
        # strm = ax.streamplot(xs,ys,vx,vy,color=v, cmap='autumn')
        # cb = fig.colorbar(strm.lines)
        # cb.set_label(r'Velocity $[m/s]$',labelpad=20)
        # ax.quiver(xs,ys,vx,vy)
        for i in range(len(xs)):
            for j in range(len(ys)):
                if v[i,j]>0.2:
                    ax.arrow(xs[i],ys[j],vx[i,j]/50,vy[i,j]/50,length_includes_head=False,head_width=0.015,head_length=0.015,width=0.003,color='orange')
                else:
                    ax.plot(xs[i],ys[j],'o',color='orange',markersize=5*v[i,j])
        ax.set_xlabel(r'$x$ $[m]$')
        ax.set_ylabel(r'$y$ $[m]$')
        ax.set_title('Wind Field')
        ax.set_xlim([0.0,self.__width])
        ax.set_ylim([0.0,self.__height])
        fig.legend(['Wind Speed'])
        if save:
            plt.savefig(f'imgs/trajectories_plots/wind-field.png',dpi=300)

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
            ax.plot(self.__tr[0,t],self.__tr[1,t],'o',color='orange',markersize=7,label='Target Distance=[{0:.2f},{0:.2f}] m'.format(self.__ex[t],self.__ey[t])) # Traget Location
            ax.plot(self.__xs[t],self.__ys[t],'bo',markersize=5) # Object Moving
            ax.quiver(self.__xs[t],self.__ys[t],self.__wind_force_x[t],self.__wind_force_y[t],scale=20,width=0.003,color='r',label='Wind Force=[{0:.2f},{0:.2f}] N'.format(self.__wind_force_x[t],self.__wind_force_y[t])) # Wind Force
            ax.quiver(self.__xs[t],self.__ys[t],self.__ctl_forces_x[t],self.__ctl_forces_y[t],scale=20,width=0.003,color="#4DBEEE",label='Control Force=[{0:.2f},{0:.2f}] N'.format(self.__ctl_forces_x[t],self.__ctl_forces_y[t])) # Control Force
            ax.plot(self.__xs[:t],self.__ys[:t],'b')
            ax.plot(self.__tr[0,:t],self.__tr[1,:t],'--',color='orange')
            # Plot fans
            for fan in self.__fans:
                ax.quiver(fan.p0[0],fan.p0[1],fan.u0[0],fan.u0[1],scale=10,color='k')
            ax.legend()

        anim = animation.FuncAnimation(fig,animation_function,frames=self.__duration,interval=1,repeat=False)

        plt.show()
        plt.close()