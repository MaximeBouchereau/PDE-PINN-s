import warnings

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from itertools import product
import statistics

import sys
import time
import datetime
from datetime import datetime as dtime

# Python code to solve 2D heat equation with PINN's for Neumann boundary conditions

class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def u0(self, x, y):
        """Initial condition function.
        Inputs:
        - x: Tensor of shape (1,n) - space variable, first dimension
        - y: Tensor of shape (1,n) - space variable, second dimension
        """
        return torch.exp(-10*(x**2+y**2))

    def __init__(self):
        zeta , HL = 256 , 3
        super().__init__()
        self.F = nn.ModuleList([nn.Linear(3, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])

    def forward(self, t, x, y):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - time variable
         - x: Tensor of shape (1,n) - space variable, first dimension
         - y: Tensor of shape (1,n) - space variable, second dimension
         """

        t, x, y = t.float().T, x.float().T, y.float().T

        X = torch.cat((t,x,y) , dim=1)

        # Structure of the solution of the equation

        for i, module in enumerate(self.F):
            X = module(X)

        return X.T

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""



    def Loss(self, t, x, y, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable, first dimension
        - y: Tensor of shape (1,n): Inputs of Neural Network - space variable, second dimension
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y
        => Returns a tensor of shape (1,1)"""

        t = torch.tensor(t, dtype=torch.float32)
        t.requires_grad = True

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        y = torch.tensor(y, dtype=torch.float32)
        y.requires_grad = True

        ones = torch.ones_like(x)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        delta_t = 1e-3  # Small parameter for finite differences in time
        delta_x = 1e-3 # Small parameter for finite differences in space, first dimension
        delta_y = 1e-3 # Small parameter for finite differences in space, second dimension

        u_hat_t = (model(t + delta_t * ones, x, y) - model(t - delta_t * ones, x, y)) / (2 * delta_t)
        u_hat_xx = (model(t, x + delta_x*ones, y) - 2*model(t, x, y) + model(t, x - delta_x*ones, y))/(delta_x**2)
        u_hat_yy = (model(t, x, y + delta_y*ones) - 2*model(t, x, y) + model(t, x,y - delta_y*ones))/(delta_y**2)
        loss_PDE = (((u_hat_t - (u_hat_xx + u_hat_yy))).abs() ** 2).mean() # Loss associated to the PDE

        theta = torch.linspace(0 , 2*np.pi , torch.numel(x)).reshape(x.size())
        u_hat_x_B = (model(t, torch.cos(theta) + delta_x * ones, torch.sin(theta)) - model(t, torch.cos(theta) - delta_x * ones, torch.sin(theta))) / (2 * delta_x)
        u_hat_y_B = (model(t, torch.cos(theta), torch.sin(theta) + delta_y * ones) - model(t, torch.cos(theta), torch.sin(theta) - delta_y * ones)) / (2 * delta_y)
        u_hat_B = - u_hat_x_B*torch.sin(theta) + u_hat_y_B*torch.cos(theta)
        loss_BC = (((u_hat_B)).abs() ** 2).mean() # Loss associated to boundary conditions (Neumann on the unit disk)

        u_hat_0 = model(0 * ones, x, y)
        u_0 = self.u0(x, y)
        loss_IC = (((u_hat_0 - u_0)).abs() ** 2).mean()  # Loss associated to initial condition

        return loss_PDE + loss_BC + loss_IC

    def Train(self, model , T = 1 , K = 1000 , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training of the model to learn solution of PDE
        Inputs:
        - model: Neural network which will be optimized
        - T: Float - Length of the time interval. Default: 1
        - K: Int - Number of data for both train and test. Default: 1000
        - BS: Int - Batch size. Default: 64
        - N_epochs: Int - Number of epochs for gradient descent. Default: 100
        - N_epochs_print: Int - Number of epochs between two prints of the Loss. Default: 10
        => Returns the lists Loss_train and Loss_test of the values of the Loss w.r.t. training and test,
        and best_model, which is the best apporoximation of the desired model"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        t_train, t_test = T * torch.rand([1, K]), T * torch.rand([1, K])
        r_train, r_test = torch.rand([1, K]), torch.rand([1, K])
        theta_train, theta_test = 2*np.pi*torch.rand([1, K]), 2*np.pi*torch.rand([1, K])
        x_train, x_test = r_train*torch.cos(theta_train), r_test*torch.cos(theta_test)
        y_train, y_test = r_train*torch.sin(theta_train), r_test*torch.sin(theta_test)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-9, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(x_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                t_batch = t_train[:, ixs]
                x_batch = x_train[:, ixs]
                y_batch = y_train[:, ixs]
                loss_train = self.Loss(t_batch, x_batch, y_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(t_test, x_test, y_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Epoch', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        torch.save((Loss_train, Loss_test, best_model, T) , "model_Heat_Equation_2D")

        pass

    def Integrate(self, name_model="model_Heat_Equation_2D",  ht = 0.1 , hx = 0.1 ,  hy = 0.1, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Poisson_Equation_2D
        - ht: Float - Step size for time. Default: 0.1
        - hx: Float - Step size for space variable, first dimension. Default: 0.1
        - hy: Float - Step size for space variable, second dimension. Default: 0.1
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        deltat = 2000*ht

        Loss_train, Loss_test, model, T = torch.load(name_model)

        t_grid, x_grid, y_grid = torch.arange(0, T+ht, ht), torch.arange(-1, 1+hx, hx), torch.arange(-1, 1+hy, hy)
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)
        theta = torch.arange(0,2*np.pi,0.01)

        # Resolution with the PINN
        z_PINN = torch.zeros([torch.numel(t_grid), grid_x.size(0), grid_x.size(1)])
        for it in range(torch.numel(t_grid)):
            for ix in range(grid_x.shape[0]):
                for iy in range(grid_y.shape[1]):
                    if grid_x[ix, iy]**2 + grid_y[ix, iy]**2 <= 1:
                        z_PINN[it, ix, iy] = model(torch.tensor([[t_grid[it]]]), torch.tensor([[grid_x[ix, iy]]]), torch.tensor([[grid_y[ix, iy]]]))
                    else:
                        z_PINN[it, ix, iy] = 0.0

        z_PINN = z_PINN.detach().numpy()

        fig = plt.figure(figsize=(5,5))
        im = plt.imshow(z_PINN[0, :, :].T, cmap="jet", aspect="equal", extent=(-1, 1, -1, 1))
        #im = plt.plot(np.cos(theta), np.sin(theta), linestyle="dashed", color="black")
        plt.title("PINN")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.colorbar()


        def animate(n):
            im.set_array(z_PINN[n, :, :].T)
            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=torch.numel(t_grid), blit=True, interval=deltat, repeat=True)


        if save_fig == False:
            plt.show()
        else:
            anim.save("PINN_Heat_equation_2D.gif", writer="pillow")
        pass