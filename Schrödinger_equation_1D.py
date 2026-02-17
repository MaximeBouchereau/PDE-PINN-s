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

# Python code to solve 1D Schrödinger equation with PINN's for Dirichlet boundary conditions

L = np.pi    # Size of the (space) domain of resolution

class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def __init__(self):
        zeta , HL = 256 , 4
        super().__init__()
        self.U = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 2, bias=True)])

    def forward(self, t, x):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - time variable
         - x: Tensor of shape (1,n) - space variable
         """

        t , x = t.float().T , x.float().T
        x_dom = torch.linspace(-L, L, 100).unsqueeze(0)

        psi = torch.cat((t,x) , dim=1)
        t_norm_psi, x_norm_psi = torch.meshgrid(t.squeeze(), x_dom.squeeze())
        norm_psi = torch.cat((t_norm_psi.reshape(torch.numel(t_norm_psi), 1), x_norm_psi.reshape(torch.numel(x_norm_psi), 1)), dim=1)

        # Structure of the solution of the equation

        for i, module in enumerate(self.U):
            psi = module(psi)
            norm_psi = module(norm_psi)

        norm_psi = norm_psi.T
        norm_psi_real = norm_psi[0, :].reshape(torch.numel(t), torch.numel(x_dom))
        norm_psi_imag = norm_psi[1, :].reshape(torch.numel(t), torch.numel(x_dom))
        norm_psi = 2 * L * torch.mean(norm_psi_real ** 2 + norm_psi_imag ** 2, dim=1).unsqueeze(0)
        psi = psi.T


        return psi / norm_psi

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""

    def u0(self, x):
        """Initial condition function.
        Inputs:
        - x: Tensor of shape (1,n): Space variable"""
        # return 0.5*torch.exp(-10*((x+L/2)/L)**2)
        return (1/torch.pi) ** 0.25 * torch.exp(-0.5 * x ** 2)

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

        delta_t = 1e-2 # Small parameter for finite differences in time
        delta_x = 1e-2 # Small parameter for finite differences in space

        u1_hat = model(t, x)[0,:]
        u2_hat = model(t, x)[1,:]
        u1_hat_t = (model(t + delta_t*ones, x)[0,:] - model(t - delta_t*ones, x)[0,:])/(2*delta_t)
        u2_hat_t = (model(t + delta_t*ones, x)[1,:] - model(t - delta_t*ones, x)[1,:])/(2*delta_t)
        u1_hat_xx = (model(t, x + delta_x*ones)[0,:] - 2*model(t, x)[0,:] + model(t, x - delta_x*ones)[0,:])/(delta_x**2)
        u2_hat_xx = (model(t, x + delta_x*ones)[1,:] - 2*model(t, x)[1,:] + model(t, x - delta_x*ones)[1,:])/(delta_x**2)

        loss_PDE = (((u1_hat_t + 0.5*u2_hat_xx - 0.5*x**2*u2_hat)).abs() ** 2).mean() + (((-u2_hat_t - 0.5*u1_hat_xx - 0.5*x**2*u1_hat)).abs() ** 2).mean() # Loss associated to the PDE

        u_hat_L, u_hat_R = model(t, -L*ones), model(t, L*ones)
        loss_BC = ((u_hat_L).abs() ** 2).mean() + ((u_hat_R).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet)

        u1_hat_0 = model(0*ones , x)[0,:]
        u2_hat_0 = model(0*ones , x)[1,:]
        u_0 = self.u0(x)
        loss_IC = (((u1_hat_0 - u_0)).abs() ** 2).mean() + ((u2_hat_0).abs() ** 2).mean() # Loss associated to initial condition
        return loss_PDE + loss_BC + loss_IC

    def Loss_autograd(self, t, x, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y.
        Derivatives are approximated with autograd.
        => Returns a tensor of shape (1,1)"""

        t = torch.tensor(t, dtype=torch.float32)
        t.requires_grad = True

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        ones = torch.ones_like(x)

        u_hat = torch.zeros_like(x)
        u_hat.requires_grad = True

        u1_hat = model(t, x)[0,:]
        u2_hat = model(t, x)[1,:]

        u1_hat_x = torch.autograd.grad(u1_hat, x, grad_outputs=torch.ones_like(u1_hat), create_graph=True)[0]
        u1_hat_xx = torch.autograd.grad(u1_hat_x, x, grad_outputs=torch.ones_like(u1_hat_x), create_graph=True)[0]

        u2_hat_x = torch.autograd.grad(u2_hat, x, grad_outputs=torch.ones_like(u2_hat), create_graph=True)[0]
        u2_hat_xx = torch.autograd.grad(u2_hat_x, x, grad_outputs=torch.ones_like(u2_hat_x), create_graph=True)[0]

        u1_hat_t = torch.autograd.grad(u1_hat, t, grad_outputs=torch.ones_like(u1_hat), create_graph=True)[0]
        u2_hat_t = torch.autograd.grad(u2_hat, t, grad_outputs=torch.ones_like(u2_hat), create_graph=True)[0]

        loss_PDE = (((u1_hat_t + 0.5 * u2_hat_xx - 0.5 * x ** 2 * u2_hat)).abs() ** 2).mean() + (((- u2_hat_t + 0.5 * u1_hat_xx - 0.5 * x ** 2 * u1_hat)).abs() ** 2).mean() # Loss associated to the PDE

        u_hat_L, u_hat_R = model(t, -L*ones), model(t, L*ones)
        loss_BC = ((u_hat_L).abs() ** 2).mean() + ((u_hat_R).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet)

        u1_hat_0 = model(0*ones , x)[0,:]
        u2_hat_0 = model(0*ones , x)[1,:]
        u_0 = self.u0(x)
        loss_IC = (((u1_hat_0 - u_0)).abs() ** 2).mean() +  ((u2_hat_0).abs() ** 2).mean() # Loss associated to initial condition

        # loss_NO = ((model(ones, x) ** 2).mean() - (u_0 ** 2).mean()) ** 2 # Constant norm
        # print("Loss_PDE:", format(loss_PDE, '.4E'), " - Loss_BC:", format(loss_BC, '.4E'), " - Loss_IC:", format(loss_IC, '.4E'))
        # print("Loss_PDE:", format(loss_PDE, '.4E'), " - Loss_BC:", format(loss_BC, '.4E'), " - Loss_IC:", format(loss_IC, '.4E'), " - Loss_NO:", format(loss_NO, '.4E'))

        return loss_PDE + loss_BC + loss_IC

    def Loss_autograd_WT(self, t, x, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable
        - model: Neural network which will be optimized
        Computes a predicted value uhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y.
        Derivatives are approximated with autograd.
        Weights are put over low time in order to ensure a correct PDE resolution
        => Returns a tensor of shape (1,1)"""

        t = torch.tensor(t, dtype=torch.float32)
        t.requires_grad = True

        x = torch.tensor(x, dtype=torch.float32)
        x.requires_grad = True

        ones = torch.ones_like(x)

        loss = torch.tensor(0.0)

        for n in range(torch.numel(t)):
            t_n = t[0, n]
            w_n = torch.tensor(0.0)
            for k in range(n):
                w_n += self.Loss_autograd(t_n * ones, x, model)
            w_n = torch.exp(-100 * w_n)
            loss += w_n * self.Loss_autograd(t_n * ones, x, model)

        return loss

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
        x_train, x_test = 2 * L * torch.rand([1, K]) - L, 2 * L * torch.rand([1, K]) - L

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

        torch.save((Loss_train, Loss_test, best_model , T) , "model_Schrodinger_Equation_1D")

        pass

    def Train_autograd(self, model , T = 1 , K = 1000 , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training of the model to learn solution of PDE
        Inputs:
        - model: Neural network which will be optimized
        - T: Float - Length of the time interval. Default: 1
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

        t_train, t_test = T * torch.rand([1, K]), T * torch.rand([1, K])
        # t_train, t_test = torch.linspace(0, T, K).unsqueeze(0), T * torch.rand([1, K])
        x_train, x_test = 2 * L * torch.rand([1, K]) - L, 2 * L * torch.rand([1, K]) - L
        # x_train, x_test = torch.linspace(-L, L, K).unsqueeze(0), torch.sort(2 * L * torch.rand([1, K]) - L)[0]

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
                loss_train = self.Loss_autograd(t_batch, x_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_train = self.Loss_autograd(t_train, t_test, model)
            loss_test = self.Loss_autograd(t_test, x_test, model)

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

        torch.save((Loss_train, Loss_test, best_model , T) , "model_Schrodinger_Equation_1D_autograd")

        pass

    def Integrate(self, name_model="model_Schrodinger_Equation_1D",  ht = 0.02 ,  hx = L/50, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Heat_Equation_1D
        - ht: Float - Step size for time. Default: 0.02
        - hx: Float - Step size for space. Default: 0.02
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        Loss_train, Loss_test, model, T = torch.load(name_model)

        t_grid, x_grid = torch.arange(0, T+ht, ht), torch.arange(-L, L+hx, hx)
        grid_t, grid_x = torch.meshgrid(t_grid, x_grid)

        # Resolution with Crank-Nicholson scheme
        print("Resolution with Crank-Nicholson scheme...")
        N , J = t_grid.shape[0] , x_grid.shape[0]
        A = -(1j)*(0.5*ht/hx**2)*(2*torch.diag(torch.ones(J-2),0) - torch.diag(torch.ones(J-3),-1) - torch.diag(torch.ones(J-3),1)) + 0.5*(1j)*ht*torch.diag(x_grid[1:-1]**2)
        z_CN = torch.zeros_like(grid_t, dtype=torch.cfloat)
        z_CN[0,:] = self.u0(x_grid)
        for n in range(N-1):
            count = int(100 * ((n+2) / N))
            sys.stdout.write("\r%d " % count + "%")
            sys.stdout.flush()
            z_CN[n+1,1:J-1] = torch.inverse(torch.eye(J-2)+0.5*A)@(torch.eye(J-2)-0.5*A)@z_CN[n,1:J-1]

        z_CN = (z_CN.real)**2 + (z_CN.imag)**2

        # Resolution with the PINN
        print(" ")
        print("Resolution with PINN...")
        z_PINN = torch.zeros_like(grid_t)
        ones = torch.ones_like(x_grid).unsqueeze(0)
        for it in range(grid_t.shape[0]):
            count = int(100 * ((it+1) / torch.numel(t_grid)))
            sys.stdout.write("\r%d " % count + "%")
            sys.stdout.flush()
            psi = model(t_grid[it] * ones, x_grid.unsqueeze(0))
            z_PINN[it, :] = psi[0, :] ** 2 + psi[1, :] ** 2
            # for ix in range(grid_x.shape[1]):
            #     z_PINN[it, ix] = model(torch.tensor([[grid_t[it, ix]]]), torch.tensor([[grid_x[it, ix]]]))[0]**2 + model(torch.tensor([[grid_t[it, ix]]]), torch.tensor([[grid_x[it, ix]]]))[1]**2

        z_PINN = z_PINN.detach().numpy()

        # Distance between both solutions
        z_DIFF = np.abs(z_CN-z_PINN)

        plt.figure(figsize=(10,10))

        plt.subplot(2, 2, 1)
        plt.imshow(z_PINN.T,cmap="jet" , aspect="auto" , extent=(0,T,-L,L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("PINN")

        plt.subplot(2, 2, 2)
        plt.imshow(z_CN.T, cmap="jet", aspect="auto", extent=(0, T, -L, L))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("Crank-Nicholson")

        plt.subplot(2, 2, 3)
        plt.imshow(z_DIFF.T, cmap="spring", aspect="auto", extent=(0, T, -L, L))
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
            plt.savefig("PINN_Schrodinger_Equation_1D.pdf" , dpi = (200))
        pass