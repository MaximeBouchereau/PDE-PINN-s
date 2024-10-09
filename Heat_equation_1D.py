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
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from itertools import product
import statistics

import sys
import time
import datetime
from datetime import datetime as dtime

# Python code to solve 1D heat equation with PINN's for Dirichlet boundary conditions

class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def __init__(self):
        zeta , HL = 128 , 2
        super().__init__()
        self.F = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])

    def forward(self, t, x):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - time variable
         - x: Tensor of shape (1,n) - space variable
         """

        t , x = t.float().T , x.float().T

        X = torch.cat((t,x) , dim=1)

        # Structure of the solution of the equation

        for i, module in enumerate(self.F):
            X = module(X)

        return X.T

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""

    def u0(self, x):
        """Initial condition function.
        Inputs:
        - x: Tensor of shape (1,n): Space variable"""
        return torch.sin(np.pi*(x+1)/2)

    def Loss(self, t, x, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y
        => Returns a tensor of shape (1,1)"""

        t = torch.tensor(t, dtype=torch.float32)
        t.requires_grad = True

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        ones = torch.ones_like(x)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        delta_t = 1e-3 # Small parameter for finite differences in time
        delta_x = 1e-3 # Small parameter for finite differences in space

        u_hat_t = (model(t + delta_t*ones, x) - model(t - delta_t*ones, x))/(2*delta_t)
        u_hat_xx = (model(t, x + delta_x*ones) - 2*model(t, x) + model(t, x - delta_x*ones))/(delta_x**2)
        #u_hat_x = (model(t, x + delta*ones) - model(t, x - delta*ones))/(2*delta)
        loss_PDE = (((u_hat_t - u_hat_xx)).abs() ** 2).mean() # Loss associated to the PDE

        u_hat_L, u_hat_R = model(t, -ones), model(t, ones)
        loss_BC = (((u_hat_L)).abs() ** 2).mean() + (((u_hat_R)).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet)

        u_hat_0 = model(0*ones , x)
        u_0 = self.u0(x)
        loss_IC = (((u_hat_0 - u_0)).abs() ** 2).mean() # Loss associated to initial condition
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

        #t_train , t_test = torch.linspace(0,T,K).reshape([1,K]) , T*torch.rand([1 , K])
        #x_train , x_test = torch.linspace(-1,1,K).reshape([1,K]) , 2*torch.rand([1 , K])-1

        t_train, t_test = T * torch.rand([1, K]), T * torch.rand([1, K])
        x_train, x_test = 2 * torch.rand([1, K]) - 1, 2 * torch.rand([1, K]) - 1

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-9, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(t_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                t_batch = t_train[:, ixs]
                x_batch = x_train[:, ixs]
                loss_train = self.Loss(t_batch, x_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(t_test, x_test, model)

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

        torch.save((Loss_train, Loss_test, best_model , T) , "model_Heat_Equation_1D")

        pass

    def Integrate(self, name_model="model_Heat_Equation_1D",  ht = 0.02 ,  hx = 0.02, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Heat_Equation_1D
        - ht: Float - Step size for time. Default: 0.02
        - hx: Float - Step size for space. Default: 0.02
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        Loss_train, Loss_test, model, T = torch.load(name_model)

        t_grid, x_grid = torch.arange(0, T+ht, ht), torch.arange(-1, 1+hx, hx)
        grid_t, grid_x = torch.meshgrid(t_grid, x_grid)

        # Resolution with Crank-Nicholson scheme
        N , J = t_grid.shape[0] , x_grid.shape[0]
        A = (0.5*ht/hx**2)*(2*torch.diag(torch.ones(J-2),0) - torch.diag(torch.ones(J-3),-1) - torch.diag(torch.ones(J-3),1))
        z_CN = torch.zeros_like(grid_t)
        z_CN[0,:] = self.u0(x_grid)
        for n in range(N-1):
            z_CN[n+1,1:J-1] = torch.inverse(torch.eye(J-2)+A)@(torch.eye(J-2)-A)@z_CN[n,1:J-1]

        # Resolution with the PINN
        z_PINN = torch.zeros_like(grid_t)
        for it in range(grid_t.shape[0]):
            for ix in range(grid_x.shape[1]):
                z_PINN[it, ix] = model(torch.tensor([[grid_t[it, ix]]]), torch.tensor([[grid_x[it, ix]]]))

        z_PINN = z_PINN.detach().numpy()

        # Distance between both solutions
        z_DIFF = np.abs(z_CN-z_PINN)

        plt.figure(figsize=(10,10))

        plt.subplot(2, 2, 1)
        plt.imshow(z_PINN.T,cmap="jet" , aspect="auto" , extent=(0,T,-1,1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("PINN")

        plt.subplot(2, 2, 2)
        plt.imshow(z_CN.T, cmap="jet", aspect="auto", extent=(0, T, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("Crank-Nicholson")

        plt.subplot(2, 2, 3)
        plt.imshow(z_DIFF.T, cmap="spring", aspect="auto", extent=(0, T, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("Difference")

        plt.subplot(2, 2, 4)
        plt.plot(list(range(len(Loss_train))), Loss_train, label="$Loss_{train}$", color="green")
        plt.plot(list(range(len(Loss_test))), Loss_test, label="$Loss_{test}$", color="red")
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss evolution")
        plt.grid()

        if save_fig == False:
            plt.show()
        else:
            plt.savefig("PINN_Heat_equation_1D.pdf" , dpi = (200))
        pass