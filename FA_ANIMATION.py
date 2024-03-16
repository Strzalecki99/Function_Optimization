#####################################################################################
#                                                                                   #
#                          FA Algorithm (Animation)                                 #
#                                                                                   #
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Firefly():
    def __init__(self, x_L_bound : int, x_R_bound : int, dimensions : int, objective_fun_num : int) -> None:
        self.dimensions = dimensions
        self.objective_fun_num = objective_fun_num
        self.x_L_bound = x_L_bound
        self.x_R_bound = x_R_bound
        self.x = np.random.uniform(x_L_bound, x_R_bound, dimensions)
        self.I = self.updateIntensity()

    def objective_function(self, x : np.ndarray) -> float:
        if self.objective_fun_num == 1:
            # Griewank Function
            return 1/40 * np.sum(np.square(x)) + 1 - np.prod(np.cos(np.radians(x / np.arange(1, np.size(x) + 1))))
        elif self.objective_fun_num == 2:
            # Ackley Function
            return -20 * np.exp(-0.2 * np.sqrt((1 / np.size(x)) * np.sum(np.square(x)))) - np.exp((1 / np.size(x)) * np.sum(np.cos(2 * np.pi * x))) + 20 + np.e
        else:
            print("Objective function not found! Returned default fuction: Sphere function")
            return np.sum(np.square(x))

    def updateIntensity(self) -> float:
        f_x = self.objective_function(self.x)
        return np.reciprocal(f_x)

    def updatePosition(self, attractivenes : float, position_difference : np.ndarray) -> None:
        random_vector = np.random.uniform(-0.5, 0.5, self.dimensions)
        self.x += attractivenes * position_difference + random_vector
        self.x = np.clip(self.x, self.x_L_bound, self.x_R_bound)
        self.I = self.updateIntensity()

class FA_Algorithm():
    def __init__(self, x_L_bound : int, x_R_bound : int, population : int, dimension : int, betha_0 : float, gamma_0 : float, stop_iterations : int, objective_fun_num : int) -> None:
        self.x_L_bound = x_L_bound
        self.x_R_bound = x_R_bound
        self.population = population
        self.dimension = dimension
        self.B_0 = betha_0
        self.G_0 = gamma_0
        self.stop_iterations = stop_iterations
        self.max_stop_iterations = stop_iterations
        self.fireflies = [Firefly(x_L_bound, x_R_bound, dimension, objective_fun_num) for i in range(population)]
        best_firefly = self.findBestFirefly()
        self.best_solution = best_firefly.objective_function(best_firefly.x)
    
    def findBestFirefly(self) -> Firefly:
        fireflies_I = [firefly.I for firefly in self.fireflies]
        index = fireflies_I.index(max(fireflies_I))
        return self.fireflies[index]
    
    def moveBestFirefly(self, best_firefly : Firefly) -> None:
        random_vector = np.random.uniform(-0.1, 0.1, best_firefly.dimensions)
        best_firefly.x += random_vector
        best_firefly.updateIntensity()

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

    def run(self):
        for i in range(self.population):
            for j in range(self.population):
                if self.fireflies[j].I > self.fireflies[i].I:
                    distance = np.linalg.norm(self.fireflies[j].x - self.fireflies[i].x)
                    attractiveness = self.B_0 * np.exp(-self.G_0 * np.square(distance))
                    position_difference = (self.fireflies[j].x - self.fireflies[i].x)
                    self.fireflies[i].updatePosition(attractiveness, position_difference)  
            
            plot_data = self.getPlotData()

        best_firefly = self.findBestFirefly()
        new_best_solution = best_firefly.objective_function(best_firefly.x)
        early_stopping = self.earlyStopping(new_best_solution)
        self.moveBestFirefly(best_firefly)
        return plot_data, early_stopping

class Animation():
    def __init__(self, solver : FA_Algorithm) -> None:
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
    solver = FA_Algorithm(x_L_bound = -40, 
                          x_R_bound = 40, 
                          population = 20, 
                          dimension = 2, 
                          betha_0 = 1.0, 
                          gamma_0 = 0.6, 
                          stop_iterations = 500, 
                          objective_fun_num = 1)
    
    animation = Animation(solver)