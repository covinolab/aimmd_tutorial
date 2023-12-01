"""
Auxiliary files for AIMMD.
"""

import numpy as np
import scipy
import torch

def shoot(x, y, inA, inB, evolve, nsteps=100, max_length=10000, D=1.0, dt=1e-5):
    """
    Perform 2-way shooting from the shooting point (x, y).
    Stop upon reaching one of the two states.
  
    Parameters
    ----------
    x, y: float, shooting point position
    inA, inB: functions, check whether a point is in A or in B
    evolve: to evolve x and y positions
    max_length: int, maximum length of a backward/forward subtrajectory
    nsteps: int, integration steps between subsequent frames
    D: float, diffusion coefficient
    dt: float, integration step
  
    Returns
    -------
    trajectory: (n, 2)-shaped numpy array
    shooting_index: position of shooting point in generated trajectory
    """
  
    # initialize backward subtrajectory at the shooting point
    backward_trajectory = np.zeros((max_length, 2)) + [x, y]
  
    # simulate until reaching one of the states
    for i in range(1, max_length):
        backward_trajectory[i] = evolve(*backward_trajectory[i - 1])
        if (inA(backward_trajectory[i, None])[0] or
            inB(backward_trajectory[i, None])[0]):
            break
  
    # extract the segment in the transition region and flip directions
    backward_trajectory = backward_trajectory[:i + 1][::-1]
  
    # initialize backward subtrajectory at the shooting point
    forward_trajectory = np.zeros((max_length, 2)) + [x, y]
  
    # simulate until reaching one of the states
    for i in range(1, max_length):
        forward_trajectory[i] = evolve(*forward_trajectory[i - 1])
        if (inA(forward_trajectory[i, None])[0] or
            inB(forward_trajectory[i, None])[0]):
            break
  
    # extract the segment in the transition region without the shooting point
    forward_trajectory = forward_trajectory[1:i+1]
  
    # join and return
    trajectory = np.append(backward_trajectory, forward_trajectory, axis=0)
    return trajectory, len(backward_trajectory) - 1


def train(model, shooting_points, shooting_results, lr=1e-4):
    """
    Trains `model` on the provided `shooting_points` and learns
    the logit-committor function of the studied system.
    
    The function uses the Adam optimizer with the log-binomial loss,
    trains a fixed number of epochs (50), and uses a default batch
    size of 4096.
    
    Parameters
    ----------
    model: PyTorch neural network model
    shooting_points: (n, d)-sized array, where n is the size of the
                     training set, and d is the dimension of the feature
                     space (input to the neural network `model`)
    shooting_results: (n, 2)-sized array; for each shooting point,
                      it contains the rA, rB numbers reporting the number
                      of times the trajectory shot from the shooting point
                      reaches state A and B, respectively
    lr: learning rate of the training
    
    Returns
    -------
    model: the PyTorch neural network model with updated parameters
    losses: a list of losses for each training epochs
    """
    batch_size = 4096
    base_epochs = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.reset_parameters()
    model.train()

    # create training set
    descriptors = torch.tensor(shooting_points, dtype=torch.float)
    results = torch.tensor(shooting_results)
    norms = torch.tensor(np.sum(shooting_results, axis=1))

    # training weights
    weights = np.zeros(len(descriptors))
    A_to_A = shooting_results[:, 0] == 2
    weights[A_to_A] = 1 / np.sum(A_to_A)
    B_to_B = shooting_results[:, 1] == 2
    weights[B_to_B] = 1 / np.sum(B_to_B)
    A_to_B = (shooting_results[:, 0] == 1) * (shooting_results[:, 1] == 1)
    weights[A_to_B] = 1 / np.sum(A_to_B)
    weights /= np.sum(weights)

    # train model
    losses = []
    epochs = base_epochs # int(len(descriptors) ** 1/3 * base_epochs)
    for i in range(epochs):

        # train cycle
        def closure():
            optimizer.zero_grad()

            # training epoch
            q = model(descriptors)
            exp_pos_q = torch.exp(+q[:, 0])
            exp_neg_q = torch.exp(-q[:, 0])
            toA_contribution = results[:, 0] * torch.log(1. + exp_pos_q)
            toB_contribution = results[:, 1] * torch.log(1. + exp_neg_q)
            loss = torch.sum((toA_contribution + toB_contribution) / norms)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        losses.append(float(loss) / len(descriptors))

    model.eval()

    return model, losses


def evaluate(model, descriptors, batch_size=4096):
    """
    Model on descriptors.
    """
    values = []
    for batch in np.array_split(descriptors, max(1, len(descriptors) // batch_size)):
        batch_values = model(torch.tensor(batch, dtype=torch.float))
        batch_values = batch_values.cpu().detach().numpy()
        batch_values = scipy.special.expit(batch_values)
        values.append(batch_values.ravel())
    return np.concatenate(values)


def selection_biases(committor_values, adaptation_bins=np.linspace(0, 1, 11)):
    """
    Determines the selection probability of each of the points of the
    provided input trajectory, such that to achieve a uniform selection
    probability in the committor space.
    
    Parameters
    ----------
    committor_values: array of committor values of the input trajectory
    adaptation_bins: the bins in the committor space in which we impose
                     the selection probability to be uniform
    
    Returns
    -------
    selection_biases: the selection probability of each trajectory point,
                      normalized to 1.
    """
  
    n_adaption_bins = len(adaptation_bins) - 1
    bin_weights = np.ones(n_adaption_bins)
    populations, _ = np.histogram(committor_values[1:-1], bins=adaptation_bins)
  
    # distribute selection biases of empty bins
    for _ in range(10):  # max recursive length
          for empty_bin_index in np.where(populations == 0)[0]:
            if empty_bin_index > 0 and empty_bin_index < n_adaption_bins - 1:
                bin_weights[empty_bin_index - 1] += bin_weights[empty_bin_index] / 2
                bin_weights[empty_bin_index + 1] += bin_weights[empty_bin_index] / 2
            elif empty_bin_index == 0:  # first bin empty
                bin_weights[empty_bin_index + 1] += bin_weights[empty_bin_index]
            else:  # last bin empty
                bin_weights[empty_bin_index - 1] += bin_weights[empty_bin_index]
            bin_weights[empty_bin_index] = 0.
  
    # compute selection biases
    selection_biases = np.zeros(len(committor_values))
    bin_indices = np.digitize(committor_values, adaptation_bins[1:-1])
    bin_indices[[0, -1]] = -1  # exclude
    for i in range(n_adaption_bins):
          selection_biases[bin_indices == i] = (
              bin_weights[i] / np.sum(bin_indices == i))
    selection_biases /= np.sum(selection_biases)
  
    return selection_biases
