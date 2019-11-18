#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import rankdata


# In[2]:


#class that defines a city
#Has a constructor function, and calculates and stores the distance between two cities
class City:
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"
    
    def distance(self, city2):
        xDist = self.x - city2.x
        yDist = self.y - city2.y
        distance = np.sqrt((xDist ** 2) + (yDist ** 2))
        return distance


# In[3]:


#another definition for distance whcih is not part of a class but can be called for two classes as arguments
def distance(city1, city2):
    xDist = city1.x - city2.x
    yDist = city1.y - city2.y
    distance = np.sqrt((xDist ** 2) + (yDist ** 2))
    return distance


# In[4]:


#a class that is used to generate different possible routes between the cities randomly and also calculated the 
#total distance and total fitness of that particular route as an inverse of the distance (since we need to 
#maximise the function(fitness) and minimize the distance hence)
class Route:
    
 
    
    def __init__(self):
        self.route = []
        self.distance = 0
        self.fitness = 0
        
    
    def CreateRoute(self,cities):
        self.route = random.sample(cities, len(cities))
    
    
    
    def RouteDistance(self):
        self.distance = 0
        for i in range(len(self.route) - 1):
            city1 = self.route[i]
            if((i + 1) >= len(self.route)):
                city2 = self.route[0]
            else:
                city2 = self.route[i + 1]
            self.distance += city1.distance(city2)

    
    def RouteFitness(self):
        
        self.fitness = 1 / float(self.distance)
    
        
    def __repr__(self):
        return (str(self.route)) 
    
    
        
        


# In[5]:


#this class holds different possible routes in a population
#performs selection, crossover, mutation and elitism to give a new generation of population

#two different methods for selection have been implemented in this, roulette wheel based selection 
#and rank based selection
#rank based selection was chosen ultimately because of better performance however
#the other method has been commented out for reference

#three different mutation methods were tried and the best one according to results has been used

#elitism is the process by which we carry forward the best chromosomes to the next generation. this depends
#on the elitesize chosen which is usually 20-25% of the initial population


class Population:
    
    def __init__(self, cities, populationSize, eliteSize, mutationRate):
        
        self.population = []
        self.popSize = populationSize
        self.eliteSize = eliteSize
        self.mutationRate = mutationRate
        
        for i in range(self.popSize):
            r = Route()
            r.CreateRoute(cities)
            self.population.append(r)
            self.population[i].RouteDistance()
            self.population[i].RouteFitness()
        
    
    def FittestSorted(self):
        
        self.population.sort(key = lambda x: x.fitness, reverse = True)
        
       
    #this is the selection based method based on roulette wheel selection
    
#     def Selection(self):
        
#         self.FittestSorted()
#         self.selectedPool = []
        
#         #based on elitesize, add them to the selection pool
#         for i in range(self.eliteSize):
#             self.selectedPool.append(self.population[i])
            
        
#         #roulette wheel selection
#         dataframe = pd.DataFrame(columns=['Route','Distance','Fitness'])
#         for i in range(self.popSize):
#             dataframe.loc[i] = [self.population[i].route,self.population[i].distance, self.population[i].fitness]
        
#         dataframe['Fitness'] = dataframe['Fitness'] / dataframe['Fitness'].sum()
#         dataframe['cumulative'] = dataframe['Fitness'].cumsum()
      
        
#         for i in range(self.popSize - self.eliteSize):
#             r = random.random()
#             for index in range(self.popSize):
#                 if(r >= dataframe.iat[index,3]):
#                     self.selectedPool.append(self.population[index])
#                     break
        
    
    #selection based on rank based algorithm
    def Selection(self):
        
        self.FittestSorted()
        self.selectedPool = []
        
        #based on elitesize, add them to the selection pool
        for i in range(self.eliteSize):
            self.selectedPool.append(self.population[i])
            
        
        
        dataframe = pd.DataFrame(columns=['Route','Distance','Fitness'])
        for i in range(self.popSize):
            dataframe.loc[i] = [self.population[i].route,self.population[i].distance, self.population[i].fitness]
        
        dataframe['Rank'] = rankdata(dataframe['Fitness'], method = 'min')
        dataframe['Rank'] = dataframe['Rank'] / dataframe['Rank'].sum()
        dataframe['cumulative'] = dataframe['Rank'].cumsum()
      
        
        for i in range(self.popSize - self.eliteSize):
            r = random.random()
            for index in range(self.popSize):
                if(r >= dataframe.iat[index,4]):
                    self.selectedPool.append(self.population[index])
                    break
        
            
        
    
    def Breed(self, parent1, parent2):
        
        gene1 = random.randrange(0, len(parent1.route), 1)
        gene2 = random.randrange(0, len(parent2.route), 1)
        
        child1 = []
        child2 = []
        child = Route()
        
        start = min(gene1, gene2)
        end = max(gene1, gene2)
        
        child1 = parent1.route[start:end + 1]
        child2 = [city for city in parent2.route if city not in child1]
        child.route = child1 + child2
        child.RouteDistance()
        child.RouteFitness()
        return child
        
    
    def Crossover(self):
        
        self.children = []
        
        breedingPool = random.sample(self.selectedPool, len(self.selectedPool))
        breedLength = self.popSize - self.eliteSize
        
        self.children = self.selectedPool[:self.eliteSize]
        
        for i in range(breedLength):
            
            child = self.Breed(breedingPool[i], breedingPool[len(breedingPool) - i - 2])
            self.children.append(child)
    
   
    
    def Mutate(self):
        
        for i in range(len(self.children)):
            
            if (random.random() > self.mutationRate):
        
                gene1 = random.randrange(0, len(self.children[i].route), 1)
                gene2 = random.randrange(0, len(self.children[i].route), 1)
                
                
                
# Method 1 of mutation (Centre Inverse Mutation):                
#                 self.children[i].route[:gene1] = self.children[i].route[:gene1][::-1]
#                 self.children[i].route[gene1:] = self.children[i].route[gene1:][::-1]

                start = min(gene1, gene2)
                end = max(gene1, gene2)

# Method 2 of mutation (Reverse Sequence Mutation) (showed the best experimental results):

                self.children[i].route[gene1:gene2] = self.children[i].route[gene1:gene2][::-1]

# Method 3 of mutation (swapping of genes):
#                 temp = self.children[i].route[gene1]
#                 self.children[i].route[gene1] = self.children[i].route[gene2]
#                 self.children[i].route[gene2] = temp

            self.children[i].RouteDistance()
            self.children[i].RouteFitness()
        
    def NextGeneration(self):
        
        self.population = self.children
        #just to be sure, calculate route distances and fitness values again
        for i in range(self.popSize):
            self.population[i].RouteDistance()
            self.population[i].RouteFitness()
        self.children = []
        
        
    def GetFittestCity(self):
        self.population.sort(key = lambda x: x.fitness, reverse = True)
        return self.population[0].distance
        
        
        


# In[6]:



#running the GA

#Randomly generate values for city
# cityList = []
# for i in range(20):
#     cityList.append(City(x = int(random.random() * 100), y = int(random.random() *100)))

cityList = []
data = pd.read_csv("test4-27603.csv", header = None)
data.head()
for i in range(data.shape[0]):
    r = data[0][i].split()
    cityList.append(City(float(r[0]),float(r[1])))

for i in range(len(cityList)):
    print(cityList[i])
    
populationSize = 100
eliteSize = 20
mutationRate = 0.02

pop = Population(cityList, populationSize, eliteSize, mutationRate)


iterations = 500
progress = []

for i in range(iterations):
    
    
    pop.Selection()
    pop.Crossover()
    pop.Mutate()
    pop.NextGeneration()
    
    
    print("Fittest city in iteration : " + str(i) + " is: " + str(pop.GetFittestCity()))
    progress.append(pop.GetFittestCity())

print(pop.population[0])
plt.plot(np.arange(iterations), progress)

plt.show()    


# In[ ]:




