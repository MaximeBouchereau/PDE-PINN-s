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

# Python code to solve 2D Poisson equation with PINN's for Dirichlet boundary conditions

class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def __init__(self):
        zeta , HL = 128 , 2
        super().__init__()
        self.F = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])

    def forward(self, x, y):
        """Structured Neural Network.
        Inputs:
         - x: Tensor of shape (1,n) - space variable, first dimension
         - y: Tensor of shape (1,n) - space variable, second dimension
         """

        x , y = x.float().T , y.float().T

        X = torch.cat((x,y) , dim=1)

        # Structure of the solution of the equation

        for i, module in enumerate(self.F):
            X = module(X)

        return X.T

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""

    def Loss(self, x, y, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable, first dimension
        - y: Tensor of shape (1,n): Inputs of Neural Network - space variable, second dimension
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y
        => Returns a tensor of shape (1,1)"""

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        y = torch.tensor(y, dtype=torch.float32)
        y.requires_grad = True

        ones = torch.ones_like(x)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        delta_x = 1e-2 # Small parameter for finite differences in space, first dimension
        delta_y = 1e-2 # Small parameter for finite differences in space, second dimension

        u_hat_xx = (model(x + delta_x*ones, y) - 2*model(x, y) + model(x - delta_x*ones, y))/(delta_y**2)
        u_hat_yy = (model(x, y + delta_y*ones) - 2*model(x, y) + model(x,y - delta_y*ones))/(delta_y**2)

        loss_PDE = (((u_hat_xx + u_hat_yy + 4*ones)).abs() ** 2).mean() # Loss associated to the PDE

        theta = torch.linspace(0 , 2*np.pi , torch.numel(x)).reshape(x.size())
        u_hat_B = model(torch.cos(theta), torch.sin(theta))
        loss_BC = (((u_hat_B)).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet on the unit disk)

        return loss_PDE + loss_BC

    def Loss_autograd(self, x, y, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable, first dimension
        - y: Tensor of shape (1,n): Inputs of Neural Network - space variable, second dimension
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y
        Derivatives are approximated with autograd
        => Returns a tensor of shape (1,1)"""

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        y = torch.tensor(y, dtype=torch.float32)
        y.requires_grad = True

        ones = torch.ones_like(x)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        u = model(x, y)

        u_hat_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_hat_xx = torch.autograd.grad(u_hat_x, x, grad_outputs=torch.ones_like(u_hat_x), create_graph=True)[0]

        u_hat_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_hat_yy = torch.autograd.grad(u_hat_y, y, grad_outputs=torch.ones_like(u_hat_y), create_graph=True)[0]

        loss_PDE = (((u_hat_xx + u_hat_yy + 4*ones)).abs() ** 2).mean() # Loss associated to the PDE

        theta = torch.linspace(0 , 2*np.pi , torch.numel(x)).reshape(x.size())
        u_hat_B = model(torch.cos(theta), torch.sin(theta))
        loss_BC = (((u_hat_B)).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet on the unit disk)

        return loss_PDE + loss_BC

    def Train(self, model , K = 1000 , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training of the model to learn solution of PDE
        Inputs:
        - model: Neural network which will be optimized
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
                x_batch = x_train[:, ixs]
                y_batch = y_train[:, ixs]
                loss_train = self.Loss(x_batch, y_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(x_test, y_test, model)

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

        torch.save((Loss_train, Loss_test, best_model) , "model_Poisson_Equation_2D")

        pass

    def Train_autograd(self, model , K = 1000 , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training of the model to learn solution of PDE
        Inputs:
        - model: Neural network which will be optimized
        - K: Int - Number of data for both train and test. Default: 1000
        - BS: Int - Batch size. Default: 64
        - N_epochs: Int - Number of epochs for gradient descent. Default: 100
        - N_epochs_print: Int - Number of epochs between two prints of the Loss. Default: 10
        Derivatives are approximated with autograd.
        => Returns the lists Loss_train and Loss_test of the values of the Loss w.r.t. training and test,
        and best_model, which is the best apporoximation of the desired model"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        #t_train , t_test = torch.linspace(0,T,K).reshape([1,K]) , T*torch.rand([1 , K])
        #x_train , x_test = torch.linspace(-1,1,K).reshape([1,K]) , 2*torch.rand([1 , K])-1

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
                x_batch = x_train[:, ixs]
                y_batch = y_train[:, ixs]
                loss_train = self.Loss_autograd(x_batch, y_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss_autograd(x_test, y_test, model)

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

        torch.save((Loss_train, Loss_test, best_model) , "model_Poisson_Equation_2D_autograd")

        pass

    def Integrate(self, name_model="model_Poisson_Equation_2D",  hx = 0.02 ,  hy = 0.02, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Poisson_Equation_2D
        - hx: Float - Step size for space variable, first dimension. Default: 0.02
        - hy: Float - Step size for space variable, second dimension. Default: 0.02
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        Loss_train, Loss_test, model = torch.load(name_model)

        x_grid, y_grid = torch.arange(-1, 1+hx, hx), torch.arange(-1, 1+hy, hy)
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)
        theta = torch.arange(0,2*np.pi,0.01)

        print("   > Computation of exact solution")
        # Computation of the exact solution
        z_Exact = torch.ones_like(grid_x) - grid_x**2 - grid_y**2
        z_Exact = z_Exact.detach().numpy()

        print("   > Computation of approximated solution by PINN's")
        # Resolution with the PINN
        z_PINN = torch.zeros_like(grid_x)
        for ix in range(grid_x.shape[0]):
            for iy in range(grid_y.shape[1]):
                z_PINN[ix, iy] = model(torch.tensor([[grid_x[ix, iy]]]), torch.tensor([[grid_y[ix, iy]]]))

        z_PINN = z_PINN.detach().numpy()

        print("   > Distance between solutions comutation")
        # Distance between both solutions
        z_DIFF = np.abs(z_Exact-z_PINN)

        print("   > L2/H1 Error computation")
        ERR_L2, ERR_H1 = [], []
        for rr in torch.linspace(0, 1, 50):
            for tt in torch.linspace(0, 2*torch.pi, 50):
                xx, yy = rr * torch.cos(tt), rr * torch.sin(tt)
                ERR_L2 += rr * (model(torch.tensor([[xx]]), torch.tensor([[yy]])) - (torch.tensor([[1 - xx ** 2 - yy ** 2]]))).abs() ** 2

                x = torch.tensor([[xx]], requires_grad=True)
                y = torch.tensor([[yy]], requires_grad=True)
                u = model(x, y)
                u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
                u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
                ERR_H1 += rr * (u_x - (torch.tensor([[- 2 * xx]]))).abs() ** 2 + rr * (u_y - (torch.tensor([[- 2 * yy]]))).abs() ** 2

        ERR_L2 = sum(ERR_L2) / len(ERR_L2)
        ERR_H1 = sum(ERR_H1) / len(ERR_H1)

        print("      - Error [L2]: ", format(ERR_L2[0] ** 0.5, '.4E'))
        print("      - Error [H1]: ", format((ERR_L2[0] + ERR_H1[0]) ** 0.5, '.4E'))

        plt.figure(figsize=(12,10))

        plt.subplot(2, 2, 1)
        plt.plot(np.cos(theta), np.sin(theta), linestyle="dashed", color="black")
        plt.imshow(z_PINN.T,cmap="jet" , aspect="equal" , extent=(-1, 1, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.title("PINN")

        plt.subplot(2, 2, 2)
        plt.plot(np.cos(theta), np.sin(theta), linestyle="dashed", color="black")
        plt.imshow(z_Exact.T, cmap="jet", aspect="equal", extent=(-1, 1, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.title("Exact solution")

        plt.subplot(2, 2, 3)
        plt.plot(np.cos(theta), np.sin(theta), linestyle="dashed", color="black")
        plt.imshow(z_DIFF.T, cmap="spring", aspect="equal", extent=(-1, 1, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
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
            plt.savefig("PINN_Poisson_equation_2D.pdf" , dpi = (200))
        pass