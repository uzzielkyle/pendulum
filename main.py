import numpy as np
import sympy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg


class Pendulum:
    def __init__(self, length, time_array, initial_angle, initial_velocity):
        """
        Initialize the Pendulum parameters.

        :param length: Length of the pendulum (meters)
        :param time_array: Array of time points (seconds)
        :param initial_angle: Initial angle (radians)
        :param initial_velocity: Initial angular velocity (rad/s)
        """
        self.length = length
        self.time_array = time_array
        self.initial_conditions = [initial_angle, initial_velocity]

    def solve_motion(self):
        """
        Solve the motion of the pendulum using Lagrangian mechanics and numerical integration.
        :return: (x, y) coordinates of the pendulum over time
        """
        # Define symbolic variables
        gravity, length, time = sp.symbols(('g', 'l', 't'))
        theta = sp.Function('theta')(time)
        dtheta = theta.diff(time)
        ddtheta = dtheta.diff(time)

        # Position in Cartesian coordinates
        x = length * sp.sin(theta)
        y = -length * sp.cos(theta)

        # Lagrange's equation of motion
        kinetic_energy = sp.Rational(
            1, 2) * (x.diff(time)**2 + y.diff(time)**2)
        potential_energy = gravity * y
        lagrangian = kinetic_energy - potential_energy
        equation_of_motion = sp.solve(
            sp.diff(lagrangian.diff(dtheta), time) -
            lagrangian.diff(theta), ddtheta
        )[0]

        # Numerical functions
        angular_acceleration_func = sp.lambdify(
            (gravity, length, theta), equation_of_motion)
        x_func = sp.lambdify((length, theta), x)
        y_func = sp.lambdify((length, theta), y)

        # Numerical integration parameters
        g = 9.81  # Acceleration due to gravity (m/s^2)
        length = self.length

        def equations(state, time, gravity, length):
            theta, angular_velocity = state
            angular_acceleration = angular_acceleration_func(
                gravity, length, theta)
            return [angular_velocity, angular_acceleration]

        # Solve ODE
        solution = odeint(equations, y0=self.initial_conditions,
                          t=self.time_array, args=(g, length))
        angles = solution[:, 0]

        # Compute x and y coordinates
        x_coords = x_func(length, angles)
        y_coords = y_func(length, angles)
        return x_coords, y_coords


# Simulation parameters
time_points = np.linspace(0, 10, 500)  # Time array (0 to 10 seconds)
pendulum = Pendulum(length=1.0, time_array=time_points,
                    initial_angle=np.pi / 4, initial_velocity=0)

# Solve for x and y positions
x_positions, y_positions = pendulum.solve_motion()

# Plot setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 0.5])
ax.set_title("Pendulum Motion")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Pendulum components: rod and smiley face
line, = ax.plot([], [], lw=3, color='brown')

# Load smiley face image
# Replace with the path to your smiley face image
smiley_image = mpimg.imread("smiley.png")
smiley_plot = ax.imshow(smiley_image, extent=[-0.1, 0.1, -0.1, 0.1], zorder=3)


def init_animation():
    """Initialize the animation with an empty frame."""
    line.set_data([], [])
    # Initial position at (0, 0)
    smiley_plot.set_extent([0, 0, 0, 0])
    return line, smiley_plot


def update_animation(frame):
    """Update the animation for each frame."""
    # Update rod position
    line.set_data([0, x_positions[frame]], [0, y_positions[frame]])
    # Update smiley position to follow the bob (not at the pivot)
    x, y = x_positions[frame], y_positions[frame]
    # Adjust size if needed
    smiley_plot.set_extent([x - 0.1, x + 0.1, y - 0.1, y + 0.1])
    return line, smiley_plot


# Animate
# Interval in milliseconds
animation_interval = 1000 * (time_points[1] - time_points[0])
pendulum_animation = animation.FuncAnimation(
    fig, update_animation, init_func=init_animation, frames=len(time_points),
    interval=animation_interval, blit=True, repeat=True
)

# Display animation
plt.show()
