# 2D Barnes-Hut Algorithm for evolution of a galaxy 

from numpy import array, empty, random, float, sqrt, exp, pi, sin, cos, tan, arctan, zeros, save, load, cross, dot
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from numpy.linalg import norm
from copy import deepcopy
from mpi4py import MPI
from tqdm import tqdm
import time

##### Simulation Parameters ###############################

# Gravitational constant in units of kpc^3 M_sun^-1 Gyr^-2
G = 4.4985022e-6

# Discrete time step.
dt = 1.e-2 #Gyr

# Theta-criterion of Barnes-Hut algorithm.
theta = 0.3

###########################################################

class Node:
    '''------------------------------------------------------------------------
    A node object will represent a body (if node.child is None) or an abstract
    node of the quad-tree if it has node.child attributes.
    ------------------------------------------------------------------------'''
    def __init__(self, m, position, momentum):
        '''----------------------------------------------------------
        Creates a child-less node using the arguments
        -------------------------------------------------------------
        .mass     : scalar m
        .position : NumPy array  with the coordinates [x,y,z]
        .momentum : NumPy array  with the components [px,py,pz]
        ----------------------------------------------------------'''
        self.m = m
        self.m_pos = m * position
        self.momentum = momentum
        self.child = None

    def position(self):
        '''----------------------------------------------------------
        Returns the physical coordinates of the node.
        ----------------------------------------------------------'''
        return self.m_pos / self.m
        
    def reset_location(self):
        '''----------------------------------------------------------
        Resets the position of the node to the 0th-order quadrant.
        The size of the quadrant is reset to the value 1.0
        ----------------------------------------------------------'''
        self.size = 1.0
        # The relative position inside the 0th-order quadrant is equal
        # to the current physical position.
        self.relative_position = self.position().copy()
        
    def place_into_quadrant(self):
        '''----------------------------------------------------------
        Places the node into next order quadrant.
        Returns the quadrant number according to the labels defined in 
        the documentation.
        ----------------------------------------------------------'''
        # The next order quadrant will have half the side of the current quadrant
        self.size = 0.5 * self.size
        return self.subdivide(1) + 2*self.subdivide(0)

    def subdivide(self, i):
        '''----------------------------------------------------------
        Places the node into the next order quadrant along the direction
        i and recalculates the relative_position of the node inside 
        this quadrant.
        ----------------------------------------------------------'''
        self.relative_position[i] *= 2.0
        quad = 0
        if self.relative_position[i] >= 1.0:
            quad = 1
            self.relative_position[i] -= 1.0
        return quad    

def add(body, node):
    '''--------------------------------------------------------------
    Defines the quad-tree by introducing a body and locating it 
    according to three conditions (see documentation for details).
    Returns the updated node containing the body.
    --------------------------------------------------------------'''
    smallest_quad = 1.e-4 # Lower limit for the side-size of the quadrants
    
    # Case 1. If node does not contain a body, the body is put in here
    new_node = body if node is None else None
    
    if node is not None and node.size > smallest_quad:
        # Case 3. If node is an external node, then the new body can not
        # be put in there. We have to verify if it has .child attribute
        if node.child is None:
            new_node = deepcopy(node)
            # Subdivide the node creating 4 children
            new_node.child = [None for i in range(4)]
            # Place the body in the appropiate quadrant
            quad = node.place_into_quadrant()
            new_node.child[quad] = node
        # Case 2. If node is an internal node, it already has .child attribute
        else:
            new_node = node

        # For cases 2 and 3, it is needed to update the mass and the position
        # of the node
        new_node.m += body.m
        new_node.m_pos += body.m_pos
        # Add the new body into the appropriate quadrant.
        quad = body.place_into_quadrant()
        new_node.child[quad] = add(body, new_node.child[quad])
    return new_node

def gravitational_force(node1, node2):
    '''--------------------------------------------------------------
    Returns the gravitational force that node1 exerts on node2.
    A short distance cutoff is introduced in order to avoid numerical
    divergences in the gravitational force.
    --------------------------------------------------------------'''
    cutoff_dist = 1.e-4
    d12 =  node1.position() - node2.position()
    d = norm(d12)
    if d < cutoff_dist:
        # Returns no Force to prevent divergences!
        return array([0., 0.])
    else:
        # Gravitational force
        return G*node1.m*node2.m*(d12)/d**3

def force_on(body, node, theta):
    '''--------------------------------------------------------------
    Barnes-Hut algorithm: usage of the quad-tree. force_on computes 
    the net force on a body exerted by all bodies in node "node".
    Note how the code is shorter and more expressive than the 
    human-language description of the algorithm.
    --------------------------------------------------------------'''
    # 1. If the current node is an external node,
    #    calculate the force exerted by the current node on b.
    if node.child is None or node.child == [None for ii in range(4)]:
        return gravitational_force(node,body)

    # 2. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal
    #    node as a single body, and calculate the force it exerts on body b.
    if node.size <norm(node.position() - body.position())*theta:
        return gravitational_force(node,body)

    # 3. Otherwise, run the procedure recursively on each child.
    return sum(force_on(body, c, theta) for c in node.child if c is not None)

def force_on_parallel(bodiesP, shoot, theta, pId, nP, comm):
    Force = []
    for bodyP in bodiesP:
        Force.append(force_on(bodyP,shoot,theta))
    dst = (pId+1)%nP
    scr = (pId-1+nP)%nP
    shootT = shoot
    shootT2 = shoot
    for jj in range(nP-1):
        if (pId%2==0):
            comm.send(shootT, dst)
            shootT = comm.recv(None, scr)
            for ii in range(len(bodiesP)):
                Force[ii] += force_on(bodiesP[ii],shootT,theta)
        else:
            shootT2 = comm.recv(None, scr)
            comm.send(shootT,dst)
            for ii in range(len(bodiesP)):
                Force[ii] += force_on(bodiesP[ii],shootT2,theta)
            shootT = shootT2
    
    """if pId == 0:
        print(Force)"""
    
    return Force
            
def verlet(bodies, root, theta, dt):
    '''--------------------------------------------------------------
    Velocity-Verlet for time evolution.
    --------------------------------------------------------------'''
    for body in bodies:
        body.momentum += 0.5*force_on(body, root, theta)*dt
        print(force_on(body, root, theta))
        body.m_pos += body.momentum*dt
        print(force_on(body, root, theta)) 
        body.momentum += 0.5*force_on(body, root, theta)*dt
        print()

def verlet_parallel(bodiesP, shoot, theta, dt, pId, nP, comm):
    '''--------------------------------------------------------------
    Velocity-Verlet method for time evolution.
    --------------------------------------------------------------'''
    Force = force_on_parallel(bodiesP, shoot, theta, pId, nP, comm)
    for ii in range(len(bodiesP)):
        bodiesP[ii].momentum += 0.5*Force[ii]*dt
        bodiesP[ii].m_pos += bodiesP[ii].momentum*dt
    Force = force_on_parallel(bodiesP, shoot, theta, pId, nP, comm)
    for ii in range(len(bodiesP)):
        bodiesP[ii].momentum += 0.5*Force[ii]*dt
        
def func(x,Distribution,Point): 
    """--------------------------------------------------------------
    Equation that follows the point of a wanted distribution that 
    matches the random one of a uniform distribution
    -----------------------------------------------------------------
       x            : Random variable in a distribution (unkonwn)
       Distribution : Distribution's function
       Point        : Random variable in the uniform distribution
    --------------------------------------------------------------"""
    return integrate.quad(Distribution,0,x)[0]-Point

def Random_numbers_distribution(f, N, x0 = 0.001, normal = False, args = None):
    """---------------------------------------
    Creates an array of N random numbers with
    a distribution density equal to f. 
    --------------------------------------"""
    if normal == False:
        norm = integrate.quad(f,0,1)[0]
        uf = lambda x: f(x)/norm #Density function with integral=1
    Uniform = random.random(N)
    Map = zeros(N)
    for ii in range(N):
        Map[ii]=fsolve(func,x0,args=(uf,Uniform[ii]))
    return Map

def kepler_galaxy(N):
    '''--------------------------------------------------------------
    Uses a uniform distrubution of masses to create a plain Disk with 
    a central Black Hole and stars orbiting around it
    -----------------------------------------------------------------
       N            : Number of particles
    --------------------------------------------------------------'''
    # Mass limits [ Solar masses]
    max_mass = 50   # Maximum mass
    min_mass = 1    # Minimum mass 
    BHM = 4e6   # Black Hole mass
    
    random.seed(10) # Seed
    # Generation of N random particles 
    status = empty([N,5])
    # Random masses varies between min_mass and max_mass in solar masses
    status[:-1, 0] = random.random(N-1)*(max_mass-min_mass) + min_mass
    #Random angle generation
    gamma = random.random(N-1)*2*pi
    init_r = 0.4 # Initial radius
    center = [0.5, 0.5] # Origin of galaxy
    #Model of density normalized
    f = lambda x: x    
    #Points mapped from the uniform distribution
    Uniform = Random_numbers_distribution(f,N-1)*init_r
    for i in range(N-1):
        #Change to cartesian coordinates
        status[i, 1] = Uniform[i]*cos(gamma[i]) + center[0]
        status[i, 2] = Uniform[i]*sin(gamma[i]) + center[1]
        # Keplerina velocity in the plain of the disc 
        Kep_v = sqrt(G*BHM/Uniform[i])
        status[i, 3] = -status[i, 0]*sin(gamma[i])*Kep_v
        status[i, 4] = status[i, 0]*cos(gamma[i])*Kep_v
    # BH's information
    status[N-1, 0] = BHM
    status[N-1, 1:3]=center
    status[N-1, 3:5]=array([0.,0.])
    return status

def bessel_galaxy(N):
    '''--------------------------------------------------------------
    Use a radial distrubution of masses which is proportional to the 
    brightness surface distributation to create a  Disk resembling an 
    spiral galaxy.
    -----------------------------------------------------------------
       N            : Number of particles
    --------------------------------------------------------------'''
    from scipy.special import kv, iv
    random.seed(10)
    init_r = 0.5  # Initial radius
    # Generates N random particles 
    status = empty([N,5])
    # Random masses varies between min_mass mass and max_mass solar masses
    max_mass = 50
    min_mass = 1
    status[:, 0] = random.random(N)*(max_mass-min_mass) + min_mass
    #Parameters of the model of density of starts (Adimentional)
    Rd = .1
    #Model of density normalized
    f = lambda x: x*exp(-x/Rd)      
    #Random angle generation
    gamma = random.random(N)*2*pi
    #Points mapped from the uniform distribution
    Map = Random_numbers_distribution(f,N, args=(Rd))*init_r
    Rd *= init_r
    center = [0.5, 0.5] # Origin of the galaxy  
    for i in range(N):
        #Change to cartesian coordinates
        status[i, 1] = Map[i]*cos(gamma[i]) + center[0]
        status[i, 2] = Map[i]*sin(gamma[i]) + center[1]
        #Velocity for particles in an exponential disc
        y = Map[i] / (2*Rd)
        sigma = sum(status[:,0])/(2*pi*(Rd**2-(init_r**2+init_r*Rd)*exp(-init_r/Rd)))
        #Magnitud
        Bessel_v = sqrt(4*pi*G*sigma*y**2*(iv(0,y)*kv(0,y)-iv(1,y)*kv(1,y)))
        status[i, 3] = -status[i, 0]*sin(gamma[i])*Bessel_v
        status[i, 4] = status[i, 0]*cos(gamma[i])*Bessel_v
    return status

def system_init_write(N, model, ini_radius, format = 'npy', data_folder = 'Data/'):
    '''--------------------------------------------------------------
    Writes a binary file with the initial state of the N-body system. 
    The latter is generated by model. It saves the data in the next
    format:
        [mi, xi, yi, zi, pxi, pyi, pzi] for i=0, ..., N-1
    --------------------------------------------------------------'''
    # Scaling of gravitational constant
    global G
    G *= (0.4/ini_radius)**3

    Numpy_Init = open(data_folder + 'Initial State.' + format, 'wb')
    state = model(N)
    save(Numpy_Init, state)
    Numpy_Init.close()

def system_init_read(N, format = 'npy', data_folder = 'Data/'):
    '''--------------------------------------------------------------
    Reads a binary file with the initial state of the N-body system.
    And builts the body class.
    --------------------------------------------------------------'''
    bodies = []
    state = load(data_folder + 'Initial State.' + format)
    for i in range(N):
       bodies.append(Node(state[i,0], state[i, 1:3], state[i, 3:5]))
    return bodies

def evolve(bodies, N, n, ini_radius, save_step, data_folder='Data/', format='npy'):
    '''--------------------------------------------------------------
    This function evolves the system in n steps of time using the 
    Verlet algorithm and the Barnes-Hut quad-tree.
    --------------------------------------------------------------'''
    # Scaling of gravitational constant
    global G
    G *= (0.4/ini_radius)**3
    File = open(data_folder + 'Evolution.' + format, 'wb')
    print('Evolution progress:')
    pbar = tqdm(total=n)
    # Principal loop over n time iterations.
    for i in range(n+1):
        # The quad-tree is recomputed at each iteration.
        start = time.process_time()   
        root = None
        for body in bodies:
            body.reset_location()
            root = add(body, root)
        #print(f"Time taken in building the tree is: {time.process_time() - start}")
        # Evolution using the Verlet method
        start = time.process_time() 
        verlet(bodies, root, theta, dt)
        #print(f"Time taken in evolution is: {time.process_time() - start}")
        # Save the data in binary files
        if i%save_step==0:
            save_data(File, bodies, N)
            pbar.update(save_step)
    File.close()
def evolve_parallel(comm, bodiesP, N, n, ini_radius, save_step, pId, nP, data_folder='Data/', format='npy', bodies = None):
    '''--------------------------------------------------------------
    This function evolves the system in n steps of time using the 
    Verlet algorithm and the Barnes-Hut quad-tree.
    --------------------------------------------------------------'''
    # Scaling of gravitational constant
    global G
    G *= (0.4/ini_radius)**3
    shoot = None
    if 0 == pId:
        File = open(data_folder + 'Evolution.' + format, 'wb')
        print('Evolution progress:')
        pbar = tqdm(total=n)
        # Principal loop over n time iterations.
        for i in range(n+1):
            # The quad-tree is recomputed at each iteration. 
            root = None
            for body in bodies:
                body.reset_location()
                root = add(body, root)
            start = time.process_time() 
            shoot = comm.scatter([root.child[ii] for ii in range(nP)], root = 0)
            # Evolution using the Verlet method
            verlet_parallel(bodiesP, shoot, theta, dt, pId, nP, comm)
            #print(f"Time taken in evolution is: {time.process_time() - start}")
            bodiesG = comm.gather(bodiesP, 0)
            bodies = []
            for b in bodiesG:
                bodies += b
            # Save the data in binary files
            if i%save_step==0:
                save_data(File, bodies, N)
                pbar.update(save_step)
        File.close()
    else: 
        # Principal loop over n time iterations.
        for i in range(n+1):
            shoot = comm.scatter(shoot, root = 0)
            verlet_parallel(bodiesP, shoot, theta, dt, pId, nP, comm)
            comm.gather(bodiesP, 0)
            
            
def save_data(File, bodies, N):
    '''--------------------------------------------------------------
    Save data of the current state of the bodies into File.
    --------------------------------------------------------------'''
    Data = zeros([N, 5])
    ii = 0
    for body in bodies:
        Data[ii, 0] = body.m
        Data[ii, 1:3] = body.position()
        Data[ii, 3:] = body.momentum
        ii +=1
    save(File, Data)

def read_evolution(N, n, save_step, data_folder='Data/', format='npy', image_folder='imagesBH/'):
    '''--------------------------------------------------------------
    Reads the evolution of the N-body system in n steps of time. 
    Creates images each save_step steps.
    --------------------------------------------------------------'''
    File = open(data_folder + 'Evolution.' + format, 'rb')
    print('Images:')
    pbar = tqdm(total=n//save_step)
    for ii in range(n//save_step):
        state = load(File)
        plot_bodies(state[:, 0], state[:, 1:3], ii, N,image_folder)
        pbar.update(1)

def plot_bodies(m, pos, i, N, image_folder='imagesBH/'):
    '''--------------------------------------------------------------
    Plots images of system's configuration.
    --------------------------------------------------------------'''
    fig = plt.figure(figsize=(10,10), facecolor= 'k')
    ax = fig.gca()
    ax.set_xlim([.08,.92])
    ax.set_ylim([.08,.92])

    # Remove grid lines
    ax.grid(False)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 

    ax.set_facecolor('k')
    
    Mmax = 50   # Maximum mass
    Mmin = 1    # Minimum mass
    dot_max = 10    # Maximum markersize
    dot_min = 0.1   # Minimun markersize
    for ii in range(N):
        if (m[ii]> Mmax):
            # Black hole
            size = 2*dot_max
            ax.scatter(pos[ii, 0], pos[ii, 1], marker='.', s=size, color='orange')
        else:
            # Stars
            size = ((dot_max-dot_min)*m[ii]+dot_min*Mmax - dot_max*Mmin)/(Mmax-Mmin)
            ax.scatter(pos[ii, 0], pos[ii, 1], marker='.', s=size, color='lightcyan')
    plt.savefig(image_folder+'bodies_{0:06}.png'.format(i))
    plt.close()

def create_video(image_folder='imagesBH/', video_name='my_video.mp4'):
    '''--------------------------------------------------------------
    Creates a .mp4 video using the stored files images
    --------------------------------------------------------------'''
    from os import listdir
    import moviepy.video.io.ImageSequenceClip
    fps = 15
    image_files = [image_folder+img for img in sorted(listdir(image_folder)) if img.endswith('.png')]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)

def tangent_velocity_distribution(N, n, ini_radius, save_step, data_folder='Data/', format='npy', image_folder='imagesBH/', imag = 1):
    '''--------------------------------------------------------------
    Reads the evolution of the N-body system in n steps of time. 
    Then, writes a binary file with tangent velocity of the bodies, 
    following next format:
        [ri, vti]   for i = 0, ..., N-1.
    --------------------------------------------------------------'''
    # Open evolution file
    File = open(data_folder + 'Evolution.' + format, 'rb')
    # Create tangent velocity file
    Velocity = open(data_folder + 'Tangent_Velocity.' + format, 'wb')
    # Number of times that data was saved
    steps = n//save_step

    print('Tangent Velocity:')
    pbar = tqdm(steps)
    # Main loop
    factor = ini_radius/0.4
    state = load(File)
    Data = tangent_velocity(state[:, 0], state[:, 1:3], state[:, 3:5], N)
    Data = array(Data)*factor
    Vmax, rmax = max(Data[1]), max(Data[0])
    for ii in range(steps):
        save(Velocity, Data, N)
        if (ii%imag==0):
            # Image's format
            fig = plt.figure(figsize=(10,10))
            plt.title(f'Tangent Velocity at t={"{:.1f}".format((ii*save_step)*0.02)} Gyr')
            ax = fig.gca()
            ax.set_xlabel('Radius, r [kpc]')
            ax.set_ylabel('Tangent Velocity, $V_t$[kpc/Gyr]')
            ax.set_xlim([0, rmax])
            ax.set_ylim([0, Vmax])
            ax.scatter(Data[0], Data[1], marker='+', color='black')
            plt.savefig(image_folder+'TV_{0:06}.png'.format(ii))
            plt.close()
        pbar.update(1)      
        state = load(File)
        Data = tangent_velocity(state[:, 0], state[:, 1:3], state[:, 3:5], N)
        Data = factor*array(Data)
    File.close()
    Velocity.close()

def tangent_velocity(m, pos, momentum, N):
    '''--------------------------------------------------------------
    Computes the tangent speed with respect to orbital plane, which 
    has normal_vector as normal vector. Returns tangent velocity and
    radius. 
    --------------------------------------------------------------'''
    r = zeros(N)
    vt = zeros(N)
    center = [0.5, 0.5]
    for i in range(N):
        vel = momentum[i] / m[i] # Velocity
        r[i] = norm(pos[i]-center)   # radius
        vt[i] = sqrt(norm(vel)**2 - dot(vel, pos[i]-center)**2 / r[i]**2)
    return r, abs(vt)

def read_model(model):
    '''--------------------------------------------------------------
    Returns the model to create galaxy according to a read input
    --------------------------------------------------------------'''
    if (model=='kepler_galaxy'):
        return kepler_galaxy
    elif (model=='bessel_galaxy'):
        return bessel_galaxy
    else:
        raise ValueError(f"Model {model} hasn't been defined.")
