#####################################################################################
#                                                                                   #
#                           GSO Algorithm (Animation)                               #
#                                                                                   #
#####################################################################################

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Firefly():
    def __init__(self, x_L_bound : int, x_R_bound : int, luciferin_decay : float, luciferin_gain : float, scope : int, dimensions : int, objective_fun_num : int) -> None:
        self.x_L_bound = x_L_bound
        self.x_R_bound = x_R_bound
        self.dimensions = dimensions
        self.objective_fun_num = objective_fun_num
        self.x = np.random.uniform(x_L_bound, x_R_bound, dimensions)
        self.luciferin = 0
        self.luciferin_decay = luciferin_decay
        self.luciferin_gain = luciferin_gain
        self.default_scope = scope
        self.adaptive_scope = scope
        self.updateLuciferin()
        
    def objective_function(self, x : np.ndarray) -> float:
        if self.objective_fun_num == 1:
            # Griewank Function
            return 1/40 * np.sum(np.square(x)) + 1 - np.prod(np.cos(np.radians(x / np.arange(1, np.size(x) + 1))))
        elif self.objective_fun_num == 2:
            # Ackley Function
            return -20 * np.exp(-0.2 * np.sqrt(np.sum(np.square(x)))) - np.exp((1 / np.size(x)) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e
        else:
            print("Objective function not found! Returned default fuction: Sphere function")
            return np.sum(np.square(x))

    def updateLuciferin(self) -> None:
        self.luciferin *= (1 - self.luciferin_decay)
        self.luciferin += np.reciprocal(self.luciferin_gain * self.objective_function(self.x))

class GSO_Algorithm():
    def __init__(self, x_L_bound : int, x_R_bound : int, population : int, luciferin_decay : float, luciferin_gain : float, scope : int, dimension : int, stop_iterations : int, step_size, betha : float, n_t : float, objective_fun_num : int) -> None:
        self.x_L_bound = x_L_bound
        self.x_R_bound = x_R_bound
        self.population = population
        self.dimension = dimension
        self.fireflies = [Firefly(x_L_bound, x_R_bound, luciferin_decay, luciferin_gain, scope, dimension, objective_fun_num) for i in range(population)]
        self.max_stop_iterations = stop_iterations
        self.step_size = step_size
        self.betha = betha
        self.n_t = n_t
        self.stop_iterations = stop_iterations
        
        best_firefly = self.findBestFirefly()
        self.best_solution = best_firefly.objective_function(best_firefly.x)

    def findBestFirefly(self) -> Firefly:
        fireflies_L = [firefly.luciferin for firefly in self.fireflies]
        index = fireflies_L.index(max(fireflies_L))
        return self.fireflies[index]
    
    def getPlotData(self) -> list:
        data = [firefly.x for firefly in self.fireflies]
        return data

    def earlyStopping(self, new_best_solution) -> bool:
        if new_best_solution < self.best_solution:
            self.best_solution = new_best_solution
            self.stop_iterations = self.max_stop_iterations
            return False
        elif  self.stop_iterations > 0:
            self.stop_iterations -= 1
            return False
        else:
            return True

    def run(self) -> float:
        for i in range(self.population):
            firefly = self.fireflies.pop(i)
            neighbours = [neighbour for neighbour in self.fireflies
                                    if np.linalg.norm(neighbour.x - firefly.x) < firefly.adaptive_scope 
                                    and neighbour.luciferin > firefly.luciferin]
            if neighbours:
                neighbours_luciferin = [neighbour.luciferin for neighbour in neighbours]
                probability = (neighbours_luciferin - firefly.luciferin) / (np.sum(neighbours_luciferin - firefly.luciferin))
                best_neighbour_index = np.argmax(probability)
                firefly.x += self.step_size * ((neighbours[best_neighbour_index].x - firefly.x) / np.linalg.norm(neighbours[best_neighbour_index].x - firefly.x))
                firefly.x = np.clip(firefly.x, self.x_L_bound, self. x_R_bound)
                firefly.updateLuciferin()
                firefly.adaptive_scope = min([firefly.default_scope, max(0, firefly.adaptive_scope + self.betha * (self.n_t - np.size(neighbours)))])
            else:
                firefly.x += np.random.uniform(-0.5, 0.5, self.dimension)
                firefly.x = np.clip(firefly.x, self.x_L_bound, self. x_R_bound)
                firefly.updateLuciferin()
            self.fireflies.insert(i, firefly)

        best_firefly = self.findBestFirefly()
        new_best_solution = best_firefly.objective_function(best_firefly.x)
        early_stopping = self.earlyStopping(new_best_solution)

        plot_data = self.getPlotData()
        return plot_data, early_stopping

class Animation():
    def __init__(self, solver : GSO_Algorithm) -> None:
        self.solver = solver
        self.animatedPlot = FuncAnimation(plt.gcf(), self.plotData, interval = 10)
        plt.tight_layout()
        plt.show()

    def plotData(self, i):
        data, pause = self.solver.run()
        plt.cla()
        for firefly in data:
            plt.plot(firefly[0], firefly[1], 'o')
        plt.grid()
        plt.xlim(self.solver.x_L_bound, self.solver.x_R_bound)
        plt.ylim(self.solver.x_L_bound, self.solver.x_R_bound)
        plt.xlabel("X_1")
        plt.ylabel("X_2")
        plt.title("Function convergence: {} \nIterations to stop: {} / {}".format(self.solver.best_solution, self.solver.stop_iterations, self.solver.max_stop_iterations), loc = 'left')
        if pause:
            self.animatedPlot.pause()
            print("Found solution: {}".format(self.solver.best_solution))

if __name__ == "__main__":
    solver = GSO_Algorithm(x_L_bound = -40, 
                           x_R_bound = 40,
                           population = 20,
                           luciferin_decay = 0.4,
                           luciferin_gain = 0.6,
                           scope = 20,
                           dimension = 2,
                           stop_iterations = 500,
                           step_size = 0.05,
                           betha = 0.08,
                           n_t = 5,
                           objective_fun_num = 1)
    
    animation = Animation(solver)