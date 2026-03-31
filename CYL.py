import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import reshape
import os

# Cauchy-Lipschitz theorem execution

# Hyperparameters
params_exp = {'name':'exponential', 'T':1, 'd':1, 'N':50, 'J':100, 'J_ref':100, 'y_0':np.array([[1.0]]), 'alpha':0.2}
params_lgt = {'name':'logistic', 'T':10, 'd':1, 'N':50, 'J':5000, 'J_ref':100, 'y_0':np.array([[0.01]]), 'alpha':0.5}
params_hos = {'name':'harmonic_oscillator', 'T':20, 'd':2, 'N':250, 'J':10000, 'J_ref':100, 'y_0':np.array([[1.0, 0.0]]), 'alpha':0.2}
params_pdl = {'name':'pendulum', 'T':20, 'd':2, 'N':250, 'J':100, 'J_ref':100, 'y_0':np.array([[2.0, 0.0]]), 'alpha':0.2}
params_lrz = {'name':'Lorenz', 'T':1.01, 'd':3, 'N':5000, 'J':100, 'J_ref':100, 'y_0':np.array([[10.0, 10.0, 20.0]]), 'alpha':1e-2}

class ODE:
    """Class for ODE tools"""

    @staticmethod
    def f(y, params):
        """Gives the dynamics of the ODE, with vector inputs.
        Inputs:
        - y: Array of shape (d, B), where B is the batch size - Space variable.
        - params: Dictionary of parameter values."""

        z = np.zeros_like(y)
        if params['name'] == "exponential":
            z = y
        if params['name'] == "logistic":
            z = y * (1 - y)
        if params['name'] == "harmonic_oscillator":
            z[0, :] = -y[1, :]
            z[1, :] = y[0, :]
        if params['name'] == "pendulum":
            z[0, :] = -y[1, :]
            z[1, :] = np.sin(y[0, :])
        if params['name'] == "Lorenz":
            sigma, rho, beta = 10, 28, 8/3
            z[0, :] = sigma * (y[1, :] - y[0, :])
            z[1, :] = rho * y[0, :] - y[1, :] - y[0, :] * y[2, :]
            z[2, :] = y[0, :] * y[1, :] - beta * y[2, :]
        return z

    @staticmethod
    def Rel_err(Y1, Y2):
        """
        Compute the relative error between a numerical solution and a reference solution.

        The function evaluates the relative error using a combination of L2 norm and
        infinity norm applied to the difference between the numerical solution and
        the reference solution.

        Parameters
        ----------
        Y1 : array-like
            Intended to represent the computed (numerical) solution. Note: this
            parameter is currently not used in the function.
        Y2 : array-like
            Intended to represent the reference solution. Note: this parameter is
            currently not used in the function.

        Returns
        -------
        float
            The relative error defined as:

            .. math::

                \\text{rel\\_err} = \\frac{\\|Y[n+1,:,:] - Y_{ref}\\|}{\\|Y_{ref}\\|}

        Notes
        -----
        - The function relies on external variables: ``Y``, ``Y_ref``, and ``n``.
        - The absolute error is computed as:
            1. L2 norm along axis 0
            2. Followed by an infinity norm
        - The same norm structure is applied to the reference solution.
        - Consider passing ``Y``, ``Y_ref``, and ``n`` as arguments for better
          modularity and clarity.

        Examples
        --------
        >>> rel_err = Rel_err(Y, Y_ref)
        >>> print(rel_err)
        """
        abs_err = np.linalg.norm(np.linalg.norm(Y1 - Y2, ord=2, axis=0), ord=np.inf, axis=0)
        norm_ref = np.linalg.norm(np.linalg.norm(Y2, ord=2, axis=0), ord=np.inf, axis=0)
        rel_err = abs_err / norm_ref
        return rel_err

    @staticmethod
    def Ref_Sol(params):
        """
        Compute a reference solution of an ODE using a fine time discretization
        and an explicit Euler scheme.

        The function integrates the dynamical system:
            y'(t) = f(y(t))
        starting from the initial condition y(0) = y_0, over a time grid that is
        refined by a factor `J_ref` compared to the coarse discretization.

        The solution is first computed on a fine grid, then subsampled to match
        the coarse grid.

        Parameters
        ----------
        params : dict
            Dictionary containing the simulation parameters:
            - 'd' : int
                Dimension of the state space.
            - 'J' : int
                Number of coarse time steps.
            - 'J_ref' : int
                Refinement factor for the time discretization.
            - 'y_0' : ndarray of shape (d,)
                Initial condition.

        Returns
        -------
        Y_ref : ndarray of shape (d, J+1)
            Reference solution evaluated on the coarse time grid.

        Notes
        -----
        - The integration is performed using the explicit Euler method
          on a refined grid of size J_ref * J.
        - The final result is obtained by subsampling every J_ref points.
        - This function is typically used to generate a high-accuracy
          reference solution for comparison with coarser schemes.
        """

        print(" > Reference solution...")
        Delta_t = params['T'] / params['J']
        Delta_t_ref = Delta_t / params['J_ref']
        Z = np.zeros((params['d'], params['J_ref'] * params['J'] + 1))
        Z[:, 0] = params['y_0']
        for j in range(params['J_ref'] * params['J'] + 1):
            print("    - " + format(100 * j / (params['J_ref'] * params['J']), '.1f') + " %", end="\r")
            Z[:, j + 1:j + 2] = Z[:, j:j + 1] + Delta_t_ref * ODE.f(Z[:, j:j + 1], params)
        Y_ref = Z[:, ::params['J_ref']]

        return Y_ref

class Thm:
    """Class for Cauchy-Lipschitz [Picard–Lindelöf] theorem illustrartion"""

    @staticmethod
    def Create_Folder(params):
        """
        Generate a descriptive folder or file name based on simulation parameters.

        This function constructs a string that encodes key parameters of the
        numerical experiment. The resulting name can be used to organize outputs
        such as plots, logs, or result files in a structured and reproducible way.

        Parameters
        ----------
        params : dict
            Dictionary containing the simulation parameters:
            - 'name' : str
                Base name of the experiment.
            - 'T' : float
                Final time of the simulation.
            - 'd' : int
                Dimension of the state space.
            - 'N' : int
                Number of iterations (e.g., Picard iterations).
            - 'J' : int
                Number of coarse time steps.
            - 'J_ref' : int
                Refinement factor for the reference solution.
            - 'y_0' : array-like
                Initial condition.
            - 'alpha' : float
                Relaxation parameter.

        Returns
        -------
        name_file : str
            A formatted string containing all parameters, suitable for use as a
            folder or file name.

        Notes
        -----
        - The function concatenates parameter values using a consistent naming
          convention: "key=value".
        - This approach is useful for tracking experiments and ensuring
          reproducibility.
        - Be cautious when using complex objects (e.g., arrays) in file names,
          as their string representation may not always be filesystem-friendly.
        """
        name_file = params['name'] + "_"
        name_file += "T=" + str(params['T']) + "_"
        name_file += "d=" + str(params['d']) + "_"
        name_file += "N=" + str(params['N']) + "_"
        name_file += "J=" + str(params['J']) + "_"
        name_file += "Jref=" + str(params['J_ref']) + "_"
        name_file += "y0=" + str(params['y_0']) + "_"
        name_file += "alpha=" + str(params['alpha'])
        return name_file

    @staticmethod
    def CYL(params, save_fig = False):
        """
            Perform Picard (Cauchy–Lipschitz / Picard–Lindelöf) iterations to solve an ODE
            and analyze the convergence toward a reference solution.

            This function computes successive approximations of the solution of the ODE:
                y'(t) = f(y(t)),  with  y(0) = y_0
            using a discretized Picard iteration scheme combined with a trapezoidal
            quadrature and a relaxation parameter.

            A high-resolution reference solution is computed beforehand and used to
            evaluate the relative error at each iteration.

            Parameters
            ----------
            params : dict
                Dictionary containing the simulation parameters:
                - 'T' : float
                    Final time.
                - 'd' : int
                    Dimension of the state space.
                - 'N' : int
                    Number of Picard iterations.
                - 'J' : int
                    Number of time steps.
                - 'J_ref' : int
                    Refinement factor for the reference solution.
                - 'y_0' : ndarray of shape (d,)
                    Initial condition.
                - 'alpha' : float
                    Relaxation parameter (0 < alpha ≤ 1).
                - 'name' : str
                    Base name for saving outputs.

            save_fig : bool, optional (default=False)
                If True, results (plots) are saved in a dedicated folder. Otherwise,
                they are displayed.

            Returns
            -------
            None

            Description
            -----------
            1. Initialization:
                - The solution tensor Y is initialized with the constant initial
                  condition over the time grid.

            2. Reference solution:
                - A fine-grid solution is computed using an explicit Euler scheme
                  (see Ref_Sol) and used as ground truth.

            3. Picard iterations:
                - At each iteration, the integral from 0 to t of f(y(s))
                is approximated using the trapezoidal rule.
                - A relaxed update is applied:
                      Y^{n+1} = alpha * Y_new + (1 - alpha) * Y^n
                  to improve stability.

            4. Error tracking:
                - The relative error with respect to the reference solution is
                  computed and stored at each iteration.

            5. Visualization:
                - The evolution of the relative error (log scale) is plotted.
                - The solution at each iteration is compared to the reference:
                    * d = 1: time series plot
                    * d = 2: phase space trajectory

            Notes
            -----
            - The method implements a discretized version of the Picard iteration
              associated with the Cauchy-Lipschitz theorem.
            - The trapezoidal rule improves the approximation of the integral compared
              to left-rectangle schemes.
            - The relaxation parameter helps control numerical instabilities and
              accelerates convergence in practice.
            - The method may require fine discretization and/or relaxation for
              stability when the Lipschitz constant or final time is large.
            """


        if save_fig:
            name_file = Thm.Create_Folder(params)
            os.makedirs(name_file, exist_ok=False)

        # Initialization
        print(" > Initial condition...")
        Y = np.zeros((params['N']+1, params['d'], params['J']+1))
        Y[0, :, :] =  params['y_0'].reshape(params['d'], 1) @ np.ones((1, params['J']+1))

        # Compute reference solution
        Y_ref = ODE.Ref_Sol(params)

        # Iterations
        print(" > Iterations of Cauchy-Lipschitz [Picard–Lindelöf]...")
        L_err = [ODE.Rel_err(Y[0, :, :], Y_ref)]
        Delta_t = params['T'] / params['J']
        for n in range(params['N']):
            F = ODE.f(Y[n, :, :], params)
            I = (Delta_t / 2) * np.cumsum(F[:, :-1] + F[:, 1:], axis=1)
            Y[n+1, :, 0:1] = Y[n, :, 0:1]
            Y_new = Y[n, :, 0:1] + I
            Y[n + 1, :, 1:] = params['alpha'] * Y_new + (1 - params['alpha']) * Y[n, :, 1:]
            rel_err = ODE.Rel_err(Y[n+1, :, :], Y_ref)
            L_err.append(rel_err)
            print("    - n = " + str(n + 1) + " / " + str(params['N']) + " - Error:", format(rel_err, '.4E'), end="\r")

        # Error evolution
        plt.figure()
        plt.plot(L_err, color="green", label=r"$\frac{||y^{[n]}-y||_{\infty}}{||y||_{\infty}}$")
        plt.yscale('log')
        plt.grid()
        plt.title("Relative Error evolution")
        plt.xlabel("Iterations")
        plt.ylabel("Relative Error")
        plt.legend()
        if save_fig:
            plt.savefig(name_file + "/Relative_error.png")
        else:
            plt.show()
        plt.close()


        # Iterations
        for n in range(params['N']+1):
            plt.figure()
            if params['d'] == 1:
                plt.plot(np.linspace(0, params['T'], params['J'] + 1), Y[n, 0, :], label="$y^{[n]}$", color="green")
                plt.plot(np.linspace(0, params['T'], params['J'] + 1), Y_ref[0, :], label="$y$", color="black", linestyle="dashed")
            if params['d'] == 2:
                plt.plot(Y[n, 0, :], Y[n, 1, :], label="$y^{[n]}$", color="green")
                plt.plot(Y_ref[0, :], Y_ref[1, :], label="$y$", color="black", linestyle="dashed")
                plt.axis('equal')
            if params['d'] == 3:
                axes = plt.axes(projection="3d")
                axes.plot(Y[n, 0, :], Y[n, 1, :], Y[n, 2, :], label="$y^{[n]}$", color="green")
                axes.plot(Y_ref[0, ], Y_ref[1, :], Y_ref[2, :], label="$y$", color="black", linestyle="dashed")
                axes.axis('equal')
            plt.legend(loc="upper right")
            plt.grid()
            plt.title("n=" + str(n) + " - Rel. error = " + str(format(L_err[n], '.4E')))
            if save_fig:
                plt.savefig(name_file + "/n=" + str(n) + ".png")
            if save_fig == False and n%5 == 0:
                plt.show()
            plt.close()

        return None