import numpy as np

minimumA = np.array([+0.21427517601479118, +1.874640867926128])
minimumB = np.array([-0.25538755928343726, -1.845107485478632])
state_radius = .5

def pes(positions):
  """
  Returns the potential energy surface of each 2D point in positions.

  Parameters
  ----------
  positions: (n, 2)-sized numpy array

  Returns
  -------
  energies: n-sized numpy array
  """

  # extract x and y vectors
  x = positions[:, 0]
  y = positions[:, 1]

  # rotate by 45 degrees
  angle = 45 / 180 * np.pi
  cos = np.cos(-angle)
  sin = np.sin(-angle)
  xr = cos * x + sin * y
  yr = cos * y - sin * x

  # apply Wolfe-Quapp potential
  energies = 2 * (xr ** 4 + yr ** 4 - 2. * xr ** 2 - 4. * yr ** 2 +
                  xr * yr + .3 * xr + .1 * yr)
  return energies


def inA(positions):
  """
  Tells if the points in positions are in state A.

  Parameters
  ----------
  positions: (n, 2)-sized numpy array

  Returns
  -------
  n-sized boolean array: `True` if in A, `False` otherwise
  """
  return np.sum((positions - minimumA) ** 2, axis=1) < state_radius ** 2


def inB(positions):
  """
  Tells if the points in positions are in state B.

  Parameters
  ----------
  positions: (n, 2)-sized numpy array

  Returns
  -------
  n-sized boolean array: `True` if in A, `False` otherwise
  """
  return np.sum((positions - minimumB) ** 2, axis=1) < state_radius ** 2


def evolve(x, y, nsteps=100, D=1.0, dt=1e-5):
  """
  Brownian motion in the WQ-potential.

  Parameters
  ----------
  positions: 2-sized numpy array, current frame positions
  nsteps: int, integration steps between subsequent frames
  D: float, diffusion coefficient
  dt: float, integration step

  Returns
  -------
  new_positions: 2-sized numpy array, next frame positions
  """

  # integration parameters
  Dt = D * dt
  noise = (2 * Dt) ** .5 * np.random.normal(size=(nsteps, 2))
  angle = 45 / 180 * np.pi
  cos = np.cos(-angle)
  sin = np.sin(-angle)

  # multiple steps between frames
  for i in range(nsteps):

    # rotate by 45 degrees
    xr = cos * x + sin * y
    yr = cos * y - sin * x

    # external force along x: fx = -dU(x,y)/dx
    fx = 2 * (- 4 * xr ** 3 * cos + 4 * yr ** 3 * sin
              + 4 * xr * cos - 8 * yr * sin
              - cos * yr + sin * xr - .3 * cos + .1 * sin)

    # external force along y: fy = -dU(x,y)/dy
    fy = 2 * (- 4 * xr ** 3 * sin - 4 * yr ** 3 * cos
              + 4 * xr * sin + 8 * yr * cos
              - sin * yr - cos * xr - .3 * sin - .1 * cos)

    # evolve (Brownian motion)
    x = x + Dt * fx + noise[i, 0]
    y = y + Dt * fy + noise[i, 1]

  # reconstruct positions
  return np.array([x, y]).T
