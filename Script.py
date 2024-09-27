import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# City and Fitness classes
class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.name) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# Algorithm functions
def createRoute(cityList):
    return random.sample(cityList, len(cityList))

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

# Final Genetic Algorithm function with visualization
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = [1 / rankRoutes(pop)[0][1]]
    print("Initial distance: " + str(progress[0]))
    
    for i in range(1, generations + 1):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
        if i % 50 == 0:
            print('Generation ' + str(i) + " Distance: " + str(progress[-1]))

    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    
    return bestRoute, progress

# GUI Class
class TSPGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Genetic Algorithm - Traveling Salesman Problem")
        self.cityList = []

        # Create GUI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Labels and Entries
        tk.Label(self.root, text="Population Size:").grid(row=0, column=0)
        self.popSize_entry = tk.Entry(self.root)
        self.popSize_entry.grid(row=0, column=1)
        self.popSize_entry.insert(tk.END, "30")

        tk.Label(self.root, text="Elite Size:").grid(row=1, column=0)
        self.eliteSize_entry = tk.Entry(self.root)
        self.eliteSize_entry.grid(row=1, column=1)
        self.eliteSize_entry.insert(tk.END, "20")

        tk.Label(self.root, text="Mutation Rate:").grid(row=2, column=0)
        self.mutationRate_entry = tk.Entry(self.root)
        self.mutationRate_entry.grid(row=2, column=1)
        self.mutationRate_entry.insert(tk.END, "0.01")

        tk.Label(self.root, text="Generations:").grid(row=3, column=0)
        self.generations_entry = tk.Entry(self.root)
        self.generations_entry.grid(row=3, column=1)
        self.generations_entry.insert(tk.END, "500")

        tk.Label(self.root, text="Number of Cities:").grid(row=4, column=0)
        self.numCities_entry = tk.Entry(self.root)
        self.numCities_entry.grid(row=4, column=1)
        self.numCities_entry.insert(tk.END, "10")

        # Buttons
        tk.Button(self.root, text="Generate Cities", command=self.generateCities).grid(row=5, column=0, pady=10)
        tk.Button(self.root, text="Run Algorithm", command=self.runAlgorithm).grid(row=5, column=1, pady=10)

        # Canvas for Matplotlib graphs
        self.fig, self.axs = plt.subplots(2)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=2)
    
    def generateCities(self):
        self.cityList = []
        numCities = int(self.numCities_entry.get())
        for i in range(0, numCities):
            self.cityList.append(City(name=i, x=int(random.random() * 200), y=int(random.random() * 200)))
        print(f"Generated {numCities} cities.")

    def runAlgorithm(self):
        popSize = int(self.popSize_entry.get())
        eliteSize = int(self.eliteSize_entry.get())
        mutationRate = float(self.mutationRate_entry.get())
        generations = int(self.generations_entry.get())

        best_route, progress = geneticAlgorithm(population=self.cityList, popSize=popSize, eliteSize=eliteSize, mutationRate=mutationRate, generations=generations)

        # Plot progress
        self.axs[0].clear()
        self.axs[0].plot(progress)
        self.axs[0].set_title('Distance vs. Generations')

        # Plot best route
        x = []
        y = []
        for city in best_route:
            x.append(city.x)
            y.append(city.y)
        x.append(best_route[0].x)
        y.append(best_route[0].y)
        self.axs[1].clear()
        self.axs[1].plot(x, y, '--o')
        self.axs[1].set_title('Best Route')
        self.canvas.draw()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGui(root)
    root.mainloop()
