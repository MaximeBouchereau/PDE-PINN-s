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

# Python code to solve controlled 1D heat equation with PINN's for Dirichlet boundary conditions and boundary control

a = 0.5

class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def __init__(self):
        zeta , HL = 256 , 4
        super().__init__()
        self.V = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])
        self.U = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])

    def forward(self, t, x):
        """Structured Neural Network.
        Inputs:
         - t: Tensor of shape (1,n) - time variable
         - x: Tensor of shape (1,n) - space variable
         """

        t, x = t.float().T, x.float().T

        v, u = torch.cat((t, x), dim=1), torch.cat((t, x), dim=1)

        # Structure of the solution/control of the PDE
        for i, module in enumerate(self.V):
            v = module(v)

        for i, module in enumerate(self.U):
            u = module(u)

        v, u = v.T, u.T
        return v, u

    def v_0(self, x):
        """Initial condition function.
        Inputs:
        - x: Tensor of shape (1,n): Space variable"""
        return torch.sin(np.pi * (x + 1) / 2)

    def v_1(self, x):
        """Final condition function.
        Inputs:
        - x: Tensor of shape (1,n): Space variable"""
        # return torch.exp(- 5 * x ** 2) * (1 - x ** 2)
        # return 0.5 * torch.sin(np.pi * x)
        # return 0.7 * torch.sin(np.pi * (x + 1) / 2) + 0.4 * torch.exp(-4 * x ** 2) * (1 - x ** 2)
        return torch.sin(np.pi * (x + 1) / 2) + 0.2 * torch.sin(2 * np.pi * (x + 1) / 2) + 0.1 * torch.sin(3 * np.pi * (x + 1) / 2)

class ML(NN):
    """Training of the neural network for solving 1D heat equation"""

    def Loss(self, t, x, T, model):
        """Computes the Loss function associated with the PINN
        Inputs:
        - t: Tensor of shape (1,n): Inputs of Neural Network - time variable
        - x: Tensor of shape (1,n): Inputs of Neural Network - space variable
        - T: Float - Final time of PDE control
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

        v_hat, u_hat = model(t, x)

        chi_omega = (torch.abs(x)<a)

        # Loss for the PDE structure
        v_hat_t = torch.autograd.grad(v_hat, t, grad_outputs=torch.ones_like(v_hat), create_graph=True)[0]
        v_hat_x = torch.autograd.grad(v_hat, x, grad_outputs=torch.ones_like(v_hat), create_graph=True)[0]
        v_hat_xx = torch.autograd.grad(v_hat_x, x, grad_outputs=torch.ones_like(v_hat_x), create_graph=True)[0]
        loss_PDE = (((v_hat_t - v_hat_xx - chi_omega * u_hat)).abs() ** 2).mean() # Loss associated to the PDE

        # Loss for Boundary Conditions
        v_hat_L, v_hat_R = model(t, -ones)[0], model(t, ones)[0]
        loss_BC = (((v_hat_L)).abs() ** 2).mean() + (((v_hat_R)).abs() ** 2).mean() # Loss associated to boundary conditions (Dirichlet)

        # Loss for Initial Condition
        v_hat_0 = model(0*ones , x)[0]
        v_0 = self.v_0(x)
        loss_IC = (((v_hat_0 - v_0)).abs() ** 2).mean() # Loss associated to initial condition

        # Loss for Final Condition
        v_hat_1 = model(T * ones, x)[0]
        v_1 = self.v_1(x)
        loss_FC = (((v_hat_1 - v_1)).abs() ** 2).mean()  # Loss associated to final condition

        return loss_PDE + loss_BC + loss_IC + loss_FC, loss_PDE, loss_BC, loss_IC, loss_FC

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

        optimizer = optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-9, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC = [], [], [], [], [] # list for loss_train values
        Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC = [], [], [], [], []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(t_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                t_batch = t_train[:, ixs]
                x_batch = x_train[:, ixs]
                loss_train, loss_train_PDE, loss_train_BC, loss_train_IC, loss_train_FC = self.Loss(t_batch, x_batch, T, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test, loss_test_PDE, loss_test_BC, loss_test_IC, loss_test_FC = self.Loss(t_test, x_test, T, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_train_PDE.append(loss_train_PDE.item())
            Loss_train_BC.append(loss_train_BC.item())
            Loss_train_IC.append(loss_train_IC.item())
            Loss_train_FC.append(loss_train_FC.item())
            Loss_test.append(loss_test.item())
            Loss_test_PDE.append(loss_test_PDE.item())
            Loss_test_BC.append(loss_test_BC.item())
            Loss_test_IC.append(loss_test_IC.item())
            Loss_test_FC.append(loss_test_FC.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Epoch', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)
                print('       > Train: PDE:', format(loss_train_PDE, '.2E'), ' - BC:', format(loss_train_BC, '.2E'), ' - IC:', format(loss_train_IC, '.2E'), ' - FC:', format(loss_train_FC, '.2E'))
                print('       > Test: PDE:', format(loss_test_PDE, '.2E'), ' - BC:', format(loss_test_BC, '.2E'), ' - IC:', format(loss_test_IC, '.2E'), ' - FC:', format(loss_test_FC, '.2E'))

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        training_time = str(datetime.timedelta(seconds=int(time.time() - start_time_train)))
        print("Computation time for training (h:min:s):", training_time)

        torch.save((Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC, Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC, best_model , T, training_time) , "model_Control_Heat_Equation_1D")

        return None

    def Print_Loss(self, name_model="model_Control_Heat_Equation_1D", save_fig=False):
        """Prints Loss value:
        - name_model: Str - Name of the trained model. Default: model_Heat_Equation_1D.
        - save_fig: Boolean - Saves the figure or not. Default: False.
        """

        Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC, Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC, model, T, train_time = torch.load(name_model)

        plt.figure()
        plt.plot(list(range(len(Loss_train))), Loss_train, label="$Loss_{train}$", color="green", linewidth=2)
        plt.plot(list(range(len(Loss_train_PDE))), Loss_train_PDE, label="$Loss_{train, PDE}$", color="forestgreen", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_train_BC))), Loss_train_BC, label="$Loss_{train, BC}$", color="limegreen", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_train_IC))), Loss_train_IC, label="$Loss_{train, IC}$", color="lime", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_train_FC))), Loss_train_FC, label="$Loss_{train, FC}$", color="aquamarine", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_test))), Loss_test, label="$Loss_{test}$", color="red", linewidth=2)
        plt.plot(list(range(len(Loss_test_PDE))), Loss_test_PDE, label="$Loss_{test, PDE}$", color="orange", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_test_BC))), Loss_test_BC, label="$Loss_{test, BC}$", color="gold", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_test_IC))), Loss_test_IC, label="$Loss_{test, IC}$", color="khaki", linestyle="dashed", linewidth=1)
        plt.plot(list(range(len(Loss_test_FC))), Loss_test_FC, label="$Loss_{test, FC}$", color="yellow", linestyle="dashed", linewidth=1)
        plt.yscale("log")
        plt.xlabel("epochs")
        # plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss evolution - Training time [h:min:s]: " + train_time)
        plt.grid()

        if save_fig == False:
            plt.show()
        else:
            plt.savefig(name_model[6:] + "_Loss_decay.pdf" , dpi = (500))
        return None

    def Integrate(self, name_model="model_Control_Heat_Equation_1D",  ht = 0.02 ,  hx = 0.02, save_fig=False):
        """Integrates the PDE with trained model.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model_Heat_Equation_1D
        - ht: Float - Step size for time. Default: 0.02
        - hx: Float - Step size for space. Default: 0.02
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        Loss_train, Loss_train_PDE, Loss_train_BC, Loss_train_IC, Loss_train_FC, Loss_test, Loss_test_PDE, Loss_test_BC, Loss_test_IC, Loss_test_FC, model, T, training_time = torch.load(name_model)

        t_grid, x_grid = torch.arange(0, T+ht, ht), torch.arange(-1, 1+hx, hx)
        grid_t, grid_x = torch.meshgrid(t_grid, x_grid)
        # u_theta = model(t_grid.unsqueeze(0), 0 * t_grid.unsqueeze(0))[1]

        # Resolution with Crank-Nicolson scheme
        print("   > Resolution with Crank-Nicolson scheme...")
        N , J = t_grid.shape[0] , x_grid.shape[0]
        A0 = (0.5 * ht / hx ** 2) * (2 * torch.diag(torch.ones(J - 2), 0) - torch.diag(torch.ones(J - 3), -1) - torch.diag(torch.ones(J - 3), 1))
        z_CN = torch.zeros_like(grid_t)
        u_theta = torch.zeros_like(grid_t)
        z_CN[0,:] = self.v_0(x_grid)
        ones = torch.ones_like(x_grid)
        u_theta[0, :] = model(0 * ones.unsqueeze(0), x_grid.unsqueeze(0))[1]
        chi_omega = (torch.abs(x_grid[1:-1]) < a)
        chi_omega_bis = (torch.abs(x_grid) < a)


        for n in range(N-1):
            count = int(100 * ((n + 2) / N))
            print(count, "%", end ="\r")
            # z_CN[n, 0] = u_theta[0, n]
            # F[0] = (0.5 * ht / hx ** 2) * (u_theta[0, n] + u_theta[0, n+1])
            # z_CN[n+1,1:J-1] = torch.inverse(torch.eye(J-2)+A)@((torch.eye(J-2)-A)@z_CN[n,1:J-1] + F)
            F = (model(t_grid[n] * ones[1:-1].unsqueeze(0), x_grid[1:-1].unsqueeze(0))[1] + model(t_grid[n] * ones[1:-1].unsqueeze(0), x_grid[1:-1].unsqueeze(0))[1]) / 2
            F = chi_omega * F.squeeze()
            u_theta[n+1, :] = chi_omega_bis * model(t_grid[n+1] * ones.unsqueeze(0), x_grid.unsqueeze(0))[1]
            A = A0
            # print(F.shape, z_CN[n+1,1:J-1].shape)
            z_CN[n+1,1:J-1] = torch.inverse(torch.eye(J-2)+A)@((torch.eye(J-2)-A)@z_CN[n,1:J-1] + ht*F)
            # z_CN[n+1, 0:J-1] = torch.inverse(torch.eye(J-1)+A)@(torch.eye(J-1)-A)@z_CN[n, 0:J-1]
        # z_CN[-1, 0] = u_theta[0, -1]

        z_CN = torch.flip(z_CN, dims=(1,))
        u_theta = torch.flip(u_theta, dims=(1,))
        z_CN = z_CN.detach().numpy()

        # Resolution with the PINN
        print("   > Resolution with PINN...")
        z_PINN = torch.zeros_like(grid_t)
        ones = torch.ones_like(x_grid).unsqueeze(0)
        for it in range(grid_t.shape[0]):
            count = int(100 * ((it + 1) / torch.numel(t_grid)))
            print(count, "%", end="\r")
            uu = model(t_grid[it] * ones, x_grid.unsqueeze(0))[0]
            z_PINN[it, :] = uu
            # for ix in range(grid_x.shape[1]):
            #     z_PINN[it, ix] = model(torch.tensor([[grid_t[it, ix]]]), torch.tensor([[grid_x[it, ix]]]))

        z_PINN = torch.flip(z_PINN, dims=(1,))
        z_PINN = z_PINN.detach().numpy()
        u_theta = u_theta.detach().numpy()

        # Distance between both solutions
        z_DIFF = np.abs(z_CN-z_PINN)

        plt.figure(figsize=(18,10))

        plt.subplot(2, 3, 1)
        plt.imshow(z_PINN.T,cmap="jet" , aspect="auto" , extent=(0,T,-1,1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("PINN")

        plt.subplot(2, 3, 2)
        plt.imshow(z_CN.T, cmap="jet", aspect="auto", extent=(0, T, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("Crank-Nicolson")

        plt.subplot(2, 3, 3)
        plt.imshow(z_DIFF.T, cmap="spring", aspect="auto", extent=(0, T, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title("Difference")

        plt.subplot(2, 3, 4)
        plt.imshow(u_theta.T, cmap="jet", aspect="auto", extent=(0, T, -1, 1))
        plt.colorbar()
        ax = plt.gca()
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x$")
        plt.title(r"$u_{\theta}$")

        plt.subplot(2, 3, 5)
        plt.plot(x_grid.detach().numpy().T, z_PINN[0, ::-1], label=r"$v_{\theta}(0, \cdot)$", color="green")
        plt.plot(x_grid.detach().numpy().T, z_CN[0, ::-1], label=r"$v_{CN}(0, \cdot)$", color="blue")
        plt.plot(x_grid.detach().numpy().T, self.v_0(x_grid), label=r"$v(0, \cdot)$", color="red")
        plt.xlabel("$t$")
        # plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title(r"$t=0$")
        plt.grid()

        plt.subplot(2, 3, 6)
        plt.plot(x_grid.detach().numpy().T, z_PINN[-1, ::-1], label=r"$v_{\theta}(T, \cdot)$", color="green")
        plt.plot(x_grid.detach().numpy().T, z_CN[-1, ::-1], label=r"$v_{CN}(T, \cdot)$", color="blue")
        plt.plot(x_grid.detach().numpy().T, self.v_1(x_grid), label=r"$v(T, \cdot)$", color="red")
        plt.xlabel("$t$")
        # plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title(r"$t=T$")
        plt.grid()

        if save_fig == False:
            plt.show()
        else:
            plt.savefig(name_model[6:] + "_PINN.pdf" , dpi = (200))
        pass