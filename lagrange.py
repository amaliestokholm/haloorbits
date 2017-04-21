import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
import matplotlib

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def matplotlib_setup():
    """ The setup, which makes nice plots for the report"""
    fig_width_pt = 328
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt
    fig_height = fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('figure', figsize=fig_size)
    matplotlib.rc('font', size=11, family='serif')
    matplotlib.rc('axes', labelsize=8)
    matplotlib.rc('legend', fontsize=8)
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    matplotlib.rc('text.latex',
                  preamble=r'\usepackage[T1]{fontenc}\usepackage{lmodern}')

matplotlib_setup()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fix_margins():
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.95)


def sphere(r):
    """ Make a sphere useable for plotting Earth or Sun in 3D plots """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear
    as spheres, cubes as cubes, etc..  
    This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')


def lagrange_points(mu):
    """ This function calculates the Lagrange points """
    L = np.zeros([5, 3])  # 5 lp with (x,y) coordinates
    # For mu << 1, we get
    L[0, 0] = 1 - (mu / 3) ** (1/3)
    L[1, 0] = 1 + (mu / 3) ** (1/3) 
    L[2, 0] = -1 + ((5 * mu) / 12)
    L[3, 0] = 0.5 - mu
    L[3, 1] = np.sqrt(3) / 2
    L[4, 0] = 0.5 - mu
    L[4, 1] = - np.sqrt(3) / 2
    return L


def three_body_a(x, mu):
    """
    This function computes the Jacobian (A) of the state given by
    x. It returns the stable and unstable eigenvectors.
    """
    # Make the different parts of A
    r1 = np.array(x[0:3])
    r1[0] += mu
    r2 = np.zeros(3)
    r2[0] = r1[0] - 1
    r2[1] = r1[1]
    r2[2] = r1[2]

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    eye = np.identity(3)
    B = (np.diag([1, 1, 0]) -
         ((1 - mu) / (r1_norm ** 3) + mu / (r2_norm ** 3)) * eye
          + (3 * (1 - mu) * (r1 * r1.T) / (r1_norm ** 5))
          + 3 * mu * (r2 * r2.T) / (r2_norm ** 5))
    
    C = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    z = np.zeros((3, 3))
    a = np.concatenate((z, eye), axis=1)
    b = np.concatenate((B, C), axis=1)
    A = np.concatenate((a, b), axis=0)
    return A
    

def three_body_odefun(x, t, mu):
    """
    This function computes the equations of motion for the CRTBP.
    x is the three-space position and velocity coordinates for the 
    particle, while mu is the mass parameter for the two large bodies.
    """
    # Make the vectors between the satellite and the bodies
    r1 = np.array(x[0:3])
    r1[0] += mu
    r2 = np.zeros(3)
    r2[0] = r1[0] - 1
    r2[1] = r1[1]
    r2[2] = r1[2]
    
    r1_norm_3 = np.linalg.norm(r1) ** 3
    r2_norm_3 = np.linalg.norm(r2) ** 3
    
    xprime = np.zeros(6)
    xprime[3:6] = ((mu - 1) * (r1 / r1_norm_3) - (mu * (r2 / r2_norm_3))
                    + ((2 * x[4]) + x[0] - (2 * x[3]) + x[1]))

    xprime[0:3] = x[3:6]
    return xprime


def three_body_ode(x, t, mu):
    mu = mu
    posvel = np.array(x[0:6])
    A = three_body_a(x=posvel, mu=mu)
    state_trans = np.reshape(x[6:], (6,6))
    dot = A * state_trans
    
    xprime = np.zeros(42)
    xprime[6:] = np.reshape(dot, (36,))
    
    # Vector between satellite and two large bodies
    r1 = np.array(x[0:3])
    r1[0] += mu
    r2 = np.zeros(3)
    r2[0] = r1[0] - 1
    r2[1] = r1[1]
    r2[2] = r1[2]
    
    r1_norm_3 = np.linalg.norm(r1) ** 3
    r2_norm_3 = np.linalg.norm(r2) ** 3
    s = np.array([2 * x[3] + x[0], -2 * x[4] + x[1], 0])
    
    xprime[3:6] = ((mu - 1) * (r1 / r1_norm_3)
                    - (mu * (r2 / r2_norm_3)) + s)

    xprime[0:3] = x[3:6]
    return xprime


def jacobi(mu, L, n=1000):
    """This function makes a contour plot of the Jacobi constant """
    def omega(x, y, mu):
        pot = (0.5 * (x ** 2 + y ** 2)
               + (1 - mu) / (np.sqrt(y ** 2 + (x + mu) ** 2))
               + mu / (np.sqrt(y ** 2 + (x + mu - 1) ** 2)))
        return pot
    
    # Initialise and make meshgrid
    xs = np.linspace(-1.5, 1.5, n)
    ys = xs
    xv, yv = np.meshgrid(xs, ys)
    
    # Make contours of potential function
    omega = omega(xv, yv, mu)
    zv = 2 * omega

    c = np.linspace(2.989, 5, 10)
    c2 = np.array([2.989, 2.994, 2.997, 3.001])
    c2 = np.concatenate((c2, np.arange(3.01, 4.9, 0.2)))
    #c2 = np.concatenate((c2, np.arange(3.5, 4.9, 0.4)))
    #c2 = np.concatenate((c2, np.arange(5.3, 10, 0.5)))
    cmap = 'Blues_r'

    # Plot
    fig, ax = plt.subplots()
    fix_margins()
    ax.set_xlabel(r'x [AU]')
    ax.set_ylabel(r'y [AU]')
    plt.plot((np.min(L[:,0]) - 0.05, np.max(L[:,0]) + 0.05), (0, 0), 'k--')
    plt.plot((0, L[3, 0]), (0, L[3,1]), 'k--')
    plt.plot((1, L[3, 0]), (0, L[3,1]), 'k--')
    plt.plot((0, L[4, 0]), (0, L[4,1]), 'k--')
    plt.plot((1, L[4, 0]), (0, L[4,1]), 'k--')
    ax.plot(0, 0, 'yo', markersize='20')
    ax.plot(1, 0, 'bo', markersize='5')
    plt.xlim([np.min(L[:,0]) - 0.05, np.max(L[:,0]) + 0.05])
    plt.ylim([np.min(L[:,1]) - 0.05, np.max(L[:,1]) + 0.05])
    con = ax.contourf(xv, yv, zv, c, cmap=cmap)
    con2 = ax.contour(xv, yv, zv, c2, colors='k')    
    ax.plot(L[:, 0], L[:, 1], 'ro', markersize='5')
    ax.annotate('L1', xy=(L[0,0], L[0,1]),color='w', 
                horizontalalignment='right', verticalalignment='bottom',)
    ax.annotate('L2', xy=(L[1,0], L[1,1]),color='w',
                    horizontalalignment='right', verticalalignment='top',)
    ax.annotate('L3', xy=(L[2,0], L[2,1]),color='w',
                    horizontalalignment='left', verticalalignment='top',)
    ax.annotate('L4', xy=(L[3,0], L[3,1]),color='w',
                    horizontalalignment='left', verticalalignment='top',)
    ax.annotate('L5', xy=(L[4,0], L[4,1]),color='w',
                    horizontalalignment='left', verticalalignment='bottom',)
    cbar = fig.colorbar(con, extend='max')
    #cbar.add_lines(con2)
    axins = zoomed_inset_axes(ax, 6, loc=2)

    # Sub region of the original image
    axins.contourf(xv, yv, zv, c, cmap=cmap)
    axins.contour(xv, yv, zv, c2, colors='k')
    x1, x2, y1, y2 = (L[0,0] - 0.04), (L[1,0] + 0.04), -0.04, 0.04
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    axins.plot(1, 0, 'bo', markersize='5')
    axins.plot(L[0:2, 0], L[0:2, 1], 'ro', markersize='5')
    axins.annotate('L1', xy=(L[0,0], L[0,1]),color='w',
                horizontalalignment='right', verticalalignment='bottom',)
    axins.annotate('L2', xy=(L[1,0], L[1,1]),color='w',
                    horizontalalignment='left', verticalalignment='top',)
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    plt.savefig('jacobi.pdf')
    plt.show()


def manifold(dx, d, y, N=51):
    """ 
    This function makes invariant manifold for CR3BP.
    dx is the size of the perturbation from Halo orbit, while 
    N is the number of trajectories
    """
    x_perturb = dx / d
    
    # the Monodromy matrix
    M = np.reshape(y[42, 6:], (6,6))
    w, v = np.linalg.eig(M)
    print(w)
    w = np.diag(w)
    real = np.array(np.imag(w) == 0)
    
    i_min = np.argmin(np.abs(w))
    i_max = np.argmax(np.abs(w))
    
    # The stable and unstable eigenvectors
    a = np.zeros(6)
    stable = v[:, a[i_min]]
    unstable = v[:, a[i_max]]

    # The stable manifold
    xs_pos = np.zeros((6, N))
    xs_neg = np.zeros((6, N))
    
    # The unstable manifold
    xu_pos = np.zeros((6, N))
    xu_neg = np.zeros((6, N))
    
    # Perturb conditions 
    for j in np.arange(N):
        # state
        phi = np.reshape(y[j, 6:], (6,6))
        
        x = y[j, 0:6].T
        
        # Go forward
        s = np.dot(phi, stable)
        u = np.dot(phi, unstable)
        s /= np.linalg.norm(s)
        u /= np.linalg.norm(u)
        
        # Perturbation
        v_perturb = x_perturb * np.linalg.norm(x[3:6]) 
        v_perturb /= np.linalg.norm(x[0:3])
        
        perturb = np.ones(6)
        perturb[0:3] *= x_perturb
        perturb[3:6] *= v_perturb

        xs_pos[:, j] = x + s * perturb
        xs_neg[:, j] = x - s * perturb
        xu_pos[:, j] = x - u * perturb
        xu_neg[:, j] = x + u * perturb
    return xs_pos, xs_neg, xu_pos, xu_neg


def main(TWO=None, THREE=None, mani=None, jac=None):
    # Define constants
    m_e = 5.972e24  # kg
    m_s = 1.989e30  # kg
    mu = m_e / (m_e + m_s)
    AU = 149597870.691
    d = AU # assume circular orbit
    r_earth = 6378  # in km
    r_sun = 695700  # in km
    
    N = 41  # number of trajectories
    dx = 1000  # size of perturbation in km
    tol = 1e-9  # tolerance

    # Initial conditions
    x1 = np.array([1.0078, 0, 0.00231796707706557, 0, 0.0118010375494954, 0])  #L2
    t1 = 3.09326571489475 * 2#* 0.33
    t2 = np.linspace(0, -3.09326571489475 * 0.3)

    state_trans_0 = np.reshape(np.identity(6), 36)
    y1 = np.concatenate((x1, state_trans_0))
    
    t1_l = np.linspace(0, t1, num=100)

    y1_n = odeint(three_body_ode, y1, t1_l, args=(mu,), rtol=tol, atol=tol)

    xs_pos, xs_neg, xu_pos, xu_neg = manifold(dx=dx, d=d, y=y1_n, N=N)
    x_n = y1_n[:, 0:6]
    
    L = lagrange_points(mu)
    L *= 1#d
    
    if mani is not None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        #ax.scatter(0, 0, 0,color='r')
        #ax.scatter(d, 0, 0, color='b')
        #ax.scatter(L[0,1],L[1,1], L[2,1], color='b')
        
        for j in np.arange(N):
            # Stable
            #d = 0.001
            #ax.set_xlim3d([1- d, 1 + (6 * d)])
            #ax.set_ylim3d([7 * d, -0.5 * d])
            #ax.set_zlim3d([-0.01, 0.03])
            y3 = odeint(three_body_odefun, xs_pos[:, j], t2, args=(mu,),
                        rtol=tol, atol=tol)
            y4 = odeint(three_body_odefun, xs_neg[:, j], t2, args=(mu,),
                        rtol=tol, atol=tol)
            #y5 = odeint(three_body_odefun, xu_pos[:, j], t2, args=(mu,),
            #            rtol=tol, atol=tol)
            #y6 = odeint(three_body_odefun, xu_neg[:, j], t2, args=(mu,),
            #            rtol=tol, atol=tol)
            ax.plot(y3[:, 0], y3[:, 1], y3[:, 2], color='y')
            ax.plot(y4[:, 0], y4[:, 1], y4[:, 2], color='r')
            #ax.plot(y5[:, 0], y5[:, 1], y5[:, 2], color='g')
            #ax.plot(y6[:, 0], y6[:, 1], y6[:, 2], color='c')
            
        plt.show()
    if THREE is not None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel('x [km]')
        ax.set_ylabel('y [km]')
        ax.set_zlabel('z [km]')
        #ax.set_aspect('equal')
    
        # make the Earth
        #xs, ys, zs = sphere(0.01)
        #xs += d
        #ax.plot_surface(xs, ys, zs, cmap='ocean')
        
        #xs_s, ys_s, zs_s = sphere(0.3)
        #ax.plot_surface(xs_s, ys_s, zs_s, cmap='hot')
        #ax.scatter(L[:, 0], L[:, 1], L[:, 2], 'b.')
        #ax.scatter(L[1, 0], L[1, 1], L[1, 2], 'ro')
        ax.plot(y1_n[:,0], y1_n[:, 1], y1_n[:, 2], 'k')
        ax.scatter(1-mu, 0,0,color='b')
        ax.scatter(L[1,0], L[1,1], L[1,2],color='r')
        #ax.plot(y2_n[:,0], y2_n[:, 1], y2_n[:, 2], 'k')
        #set_axes_equal(ax)
    if TWO is not None:
        fig, ax = plt.subplots()
        fix_margins()
        ax.set_xlabel('x [AU]')
        ax.set_ylabel('y [AU]')
        ax.plot(0, 0, 'ro', markersize='50')
        ax.plot(d, 0, 'bo', markersize='7')
        plt.xlim([np.min(L[:,0]) - 0.05, np.max(L[:,0]) + 0.05])
        ax.plot(L[:, 0], L[:, 1], 'ko', markersize='7')
        phi = np.linspace(0, 2 * np.pi)
        ax.axis('equal')
        ax.plot(d * np.cos(phi), d * np.sin(phi), 'k-')
        axins = zoomed_inset_axes(ax, 6, loc=1)

        # Sub region of the original image
        axins.plot(d * np.cos(phi), d * np.sin(phi), 'k-')
        x1, x2, y1, y2 = (L[0,0] - 0.04 * d), (L[1,0] + 0.04 * d), -0.04 *d, 0.04 *d
        axins.axis('equal')
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        axins.plot(d, 0, 'bo', markersize='7')
        axins.plot(L[0:2, 0], L[0:2, 1], 'ko', markersize='7')
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    if jac is not None:
        jacobi(mu=mu, L=L)
    plt.show()

main(jac=1)
