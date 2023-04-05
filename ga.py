import random
import pickle
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

custom_objects = {"RBFLayer": RBFLayer}
with tensorflow.keras.utils.custom_object_scope(custom_objects):
    model = load_model('rbf1.h5')
    model2 = load_model('rbf2.h5')
    model3 = load_model('rbf3.h5')
    model4 = load_model('rbf4.h5')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.uniform, 0.5, 1.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 36)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    return 1,

def feasible(individual):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    for i in range(8):
        if (0.5>individual[i] or individual[i]>1.5):
            return False
    return True

def distance(individual):
    """A distance function to the feasibility region."""
    return sum((np.array(individual)-1)**2)

toolbox.register("evaluate", evalOneMax)
toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, 0, distance))

toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=2, low=0.5, up=1.5)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=2, low=0.5, up=1.5, indpb=1)
toolbox.register("select", tools.selTournament, tournsize=3)

#----------
ps = 50

def main():
    random.seed(64)

    pop = toolbox.population(n=ps)
    print("Start of evolution")
    
    fitnesses = []


    for i in range(ps):
        sum = 0
        for j in range(ps):
            if(i!=j):
                sum += model.predict(np.concatenate((pop[i],pop[j])).reshape(1,72))[0][0]
                sum += model2.predict(np.concatenate((pop[i],pop[j])).reshape(1,72))[0][0]
                sum += model3.predict(np.concatenate((pop[i],pop[j])).reshape(1,72))[0][0]
                sum += model4.predict(np.concatenate((pop[i],pop[j])).reshape(1,72))[0][0]
                sum /= 4
        fitnesses.append(tuple([sum]))


    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while g < 101:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.9, mutpb=0.2)

        
        fitnesses = []

        for i in range(ps):
            sum = 0
            for j in range(ps):
                if(i!=j):
                    sum += model.predict(np.concatenate((offspring[i],offspring[j])).reshape(1,16))[0][0]
                    sum += model2.predict(np.concatenate((pop[i],pop[j])).reshape(1,16))[0][0]
                    sum += model3.predict(np.concatenate((pop[i],pop[j])).reshape(1,16))[0][0]
                    sum += model4.predict(np.concatenate((pop[i],pop[j])).reshape(1,16))[0][0]
                    sum /= 4
            fitnesses.append(tuple([sum]))
            print(sum)
        
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = np.array([ind.fitness.values[0] for ind in pop])

        print(fits)

        length = len(pop)
        mean = np.sum(fits) / length


        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)


        s = str(min(fits)) + ',' + str(max(fits)) + ',' + str(mean) + '\n'
        f=open("log.txt","a+")
        f.writelines(s)
        f.close()

        best_ind = tools.selBest(pop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        l1 = []
        for i in pop:
            l2 = []
            for j in i:
                l2.append(j)
            l1.append(l2)

        with open('pop' + str(g) + '.pkl','wb') as f:
            pickle.dump(l1, f)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()

