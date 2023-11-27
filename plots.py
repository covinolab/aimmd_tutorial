import scipy
import numpy as np
import matplotlib.pyplot as plt

def plot_committor_values(committor_values, biases=None):

    if biases is None:
        biases = np.ones(len(committor_values))
    
    fig = plt.figure(figsize=(4,3))
    plt.plot(committor_values, ':', color='black')
    plt.scatter(np.arange(len(committor_values)), committor_values,
                s = biases / np.sum(biases) * 5000,
                color='dodgerblue', alpha=.5, zorder=10, label='selection bias')
    plt.grid()
    plt.xlabel('Frame index')
    plt.ylabel('Committor value')
    plt.gca().set_yticks(np.linspace(0, 1, 11))
    plt.legend()
    return fig

def plot_pes(X, Y, V, A=None, B=None, P=None,
             limits=np.array([-2.25, +2.25, -2.25, +2.25])):
    """
    # states and committor
    """
    
    keepers = ((X >= limits[0, None]) * (X <= limits[1, None]) *
               (Y >= limits[None, 2]) * (Y <= limits[None, 3]))
    n = np.sum((X[0, :] >= limits[0]) * (X[0, :] <= limits[1]))
    X = X[keepers].reshape((-1, n))
    Y = Y[keepers].reshape((-1, n))
    V = V[keepers].reshape((-1, n))

    plt.figure(figsize=(4,3))
    plt.contourf(X, Y, V, levels=np.linspace(-14, 4, 19), cmap='magma')
    plt.colorbar(label='Energy [$k_BT$]')
    
    plt.gca().set_aspect('equal')

    if A is not None:
        A = A[keepers].reshape((-1, n))
        minimumA = [np.mean(X[A]), np.mean(Y[A])]
        plt.contourf(X, Y, A, levels=[0.5, 1.5], colors='dodgerblue')
        plt.contour(X, Y, A, levels=[0.5, 1.5], colors='black', linewidths=1)
        plt.text(*minimumA, 'A', ha='center', va='center', fontsize=20)

    if B is not None:
        B = B[keepers].reshape((-1, n))
        minimumB = [np.mean(X[B]), np.mean(Y[B])]
        plt.contourf(X, Y, B, levels=[0.5, 1.5], colors='tomato')
        plt.contour(X, Y, B, levels=[0.5, 1.5], colors='black', linewidths=1)
        plt.text(*minimumB, 'B', ha='center', va='center', fontsize=20)

    if P is not None:
        P = P[keepers].reshape((-1, n))
        contour = plt.contour(X, Y, P, levels=np.linspace(.1, .9, 9),
                              colors='#777777', linewidths=1)
        plt.clabel(contour, colors='black')

    return plt.gca()


def plot_model(model, X, Y, V, A, B, P,
               shooting_points=None, shooting_results=None,
               limits=np.array([-2.25, +2.25, -2.25, +2.25])):
    """
    # states and committor
    """
    
    keepers = ((X >= limits[0, None]) * (X <= limits[1, None]) *
               (Y >= limits[None, 2]) * (Y <= limits[None, 3]))
    n = np.sum((X[0, :] >= limits[0]) * (X[0, :] <= limits[1]))
    X = X[keepers].reshape((-1, n))
    Y = Y[keepers].reshape((-1, n))
    V = V[keepers].reshape((-1, n))
    A = A[keepers].reshape((-1, n))
    B = B[keepers].reshape((-1, n))
    P = P[keepers].reshape((-1, n))

    plt.figure(figsize=(4,3))
    V[A] = np.nan
    V[B] = np.nan
    plt.contour(X, Y, V, levels=np.linspace(-14, 4, 19),
              colors='black', alpha=.33, zorder=10)
    plt.xlim(-2.25, 2.25)
    plt.ylim(-2.25, 2.25)
    plt.gca().set_aspect('equal')

    # visualize states
    plt.contourf(X, Y, A, levels=[0.5, 1.5], colors='dodgerblue')
    plt.contour(X, Y, A, levels=[0.5, 1.5], colors='black', linewidths=1)
    minimumA = [np.mean(X[A]), np.mean(Y[A])]
    minimumB = [np.mean(X[B]), np.mean(Y[B])]
    plt.text(*minimumA, 'A', ha='center', va='center', fontsize=20)
    plt.contourf(X, Y, B, levels=[0.5, 1.5], colors='tomato')
    plt.contour(X, Y, B, levels=[0.5, 1.5], colors='black', linewidths=1)
    plt.text(*minimumB, 'B', ha='center', va='center', fontsize=20)

    # visualize committor model
    xy = np.array([X.ravel(), Y.ravel()]).T
    values = evaluate(model, xy)
    print(xy.shape, X.shape, values.shape)
    contour = plt.contour(X, Y, values.reshape(X.shape),
                        levels=np.linspace(.1, .9, 9),
                        colors='#777777', linewidths=1, zorder=-10)

    # true vs committor model
    deviation = values.reshape(X.shape) - P
    deviation[V > 4] = np.nan
    plt.contourf(X, Y, deviation,
                        levels=np.linspace(-.15, .15, 7),
                        cmap='Spectral', linewidths=1, zorder=-10)
    plt.colorbar(label='Estimated - true $p_B$')

    if shooting_points is not None:
        if shooting_results is not None:
            c = ['chartreuse' if r[0] == 1 and r[1] == 1 else
                 'cyan' if r[0] == 2 else
                 'orange' if r[1] == 2 else
                 'gray' for r in shooting_results]
    else:
        c = 'white'

    # visualize training set
    plt.scatter(*shooting_points.T, s=7, c='black', zorder=100)
    plt.scatter(*shooting_points.T, s=5, c=c, zorder=102)

    return plt.gca()

def plot_energy_profile(values, weights,
                        reference_values, reference_weights):
    vmin = max(-25, scipy.special.logit(np.min(values)))
    vmax = min(+25, scipy.special.logit(np.max(values)))
    bins = scipy.special.expit(np.linspace(vmin, vmax, 51))
    central_values = (bins[1:] + bins[:-1]) / 2
    plt.figure(figsize=(4,3))

    # reference
    populations = np.histogram(reference_values, bins,
                            weights=reference_weights)[0]
    free_energy = -np.log(populations / np.sum(populations))
    plt.plot(central_values, free_energy, ':', color='black', label='reference')

    # estimates
    populations = np.histogram(values, bins, weights=weights)[0]
    free_energy = -np.log(populations / np.sum(populations))
    plt.plot(central_values, free_energy, color='dodgerblue', lw=2)

    plt.legend()
    plt.grid()
    plt.gca().set_xscale('logit')
    plt.xlabel('Committor model')
    plt.ylabel('Free energy [$k_BT$]')
    return plt.gca()

def plot_crossing_probability(extremes, weights):
    figure = plt.figure(figsize=(4,3))
    order = np.argsort(extremes)[::-1]
    plt.barh(np.append([0.], np.cumsum(weights[order])[:-1]),
          extremes[order],
          weights[order],
          align='edge', color='dodgerblue',
          linewidth=.5, edgecolor='black', zorder=4)
    plt.xlim(0, 1)
    plt.grid()
    plt.xlabel('Extremes')
    plt.ylabel('Weights')

    x = np.linspace(0, 1, 101)
    plt.plot(x, 1/x, 'r', zorder=5)
    plt.ylim(0, min(np.sum(weights), 10))
    return plt.gca()
  
def plot_2d_energy_profile(configurations, weights, X, Y, A=None, B=None, P=None, limits=np.array([-2.25, +2.25, -2.25, +2.25])):
    
    keepers = ((X >= limits[0, None]) * (X <= limits[1, None]) *
               (Y >= limits[None, 2]) * (Y <= limits[None, 3]))
    n = np.sum((X[0, :] >= limits[0]) * (X[0, :] <= limits[1]))
    X = X[keepers].reshape((-1, n))
    Y = Y[keepers].reshape((-1, n))
    
    figure = plt.figure(figsize=(4, 3))
    bins = [[-np.inf] + list(X[0, :]),
      list(Y[:, 0]) + [+np.inf]]
    populations = np.histogramdd(configurations, bins, weights=weights)[0].T
    V = -np.log(populations)
    V -= V[len(X) // 2, len(Y) // 2]
    plt.contourf(X, Y, V, levels=np.linspace(-14, 4, 19), cmap='magma')
    plt.colorbar(label='Energy [$k_BT$]')
    plt.xlim(-2.25, 2.25)
    plt.ylim(-2.25, 2.25)
    plt.gca().set_aspect('equal')

    if A is not None:
        A = A[keepers].reshape((-1, n))
        minimumA = [np.mean(X[A]), np.mean(Y[A])]
        plt.contourf(X, Y, A, levels=[0.5, 1.5], colors='dodgerblue')
        plt.contour(X, Y, A, levels=[0.5, 1.5], colors='black', linewidths=1)
        plt.text(*minimumA, 'A', ha='center', va='center', fontsize=20)

    if B is not None:
        B = B[keepers].reshape((-1, n))
        minimumB = [np.mean(X[B]), np.mean(Y[B])]
        plt.contourf(X, Y, B, levels=[0.5, 1.5], colors='tomato')
        plt.contour(X, Y, B, levels=[0.5, 1.5], colors='black', linewidths=1)
        plt.text(*minimumB, 'B', ha='center', va='center', fontsize=20)

    if P is not None:
        P = P[keepers].reshape((-1, n))
        contour = plt.contour(X, Y, P, levels=np.linspace(.1, .9, 9),
                      colors='#777777', linewidths=1)
        plt.clabel(contour, colors='black')

    return plt.gca()

