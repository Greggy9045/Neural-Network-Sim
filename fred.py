import numpy as np 
import random
import os
import time
import math
from threading import Timer
import json

width, height = 600, 800
worldsize = min(width, height)

target_x, target_y = worldsize / 2, worldsize / 2
end_x = 0
end_y = 0
FredGeneration = 0
currFredGeneration = FredGeneration - 1 
start_time = time.time()

class FRED:
    def __init__(self):
        self.xPOS, self.yPOS = random.uniform(0, worldsize), random.uniform(0, worldsize)
        self.step = 5
        self.age = 0 
        self.base_energy = 300
        self.moveEnergy = 5 
        self.energy = self.base_energy * 10 # number variable is the time in seconds
        self.PathMemory = []
        self.visited_positions  = set()
        self.visitedradius = 15 

        self.weights = np.random.randn(2, 4) * 0.1
        self.biases = np.random.randn(4) * 0.1

        self.old_step = self.step
        self.old_base_energy = self.base_energy
        self.old_weights = self.weights
        self.old_biases = self.biases

    def FORWARD(self, X):
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def has_visited(self, x, y):
        return (round(x), round(y)) in self.visited_positions
    
    def move(self):
        self.age += 1
        self.energy -= self.moveEnergy 

        self.visited_positions.add((round(self.xPOS), round(self.yPOS)))
        old_dist = self.intelect()
        X = np.array([self.xPOS / worldsize, 
                    self.yPOS / worldsize,])

        output = self.FORWARD(X[:2])
        prob = self.softmax(output)

        directions = [
            (0, -self.step),
            (0, self.step), 
            (-self.step, 0),  
            (self.step, 0)  
        ]
        exploration_bonus = np.zeros(4)
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = self.xPOS + dx, self.yPOS + dy
            if (round(new_x), round(new_y)) not in self.visited_positions:  # Fixed set check
                exploration_bonus[i] = 0.1
        
        prob = prob * 0.7 + exploration_bonus * 0.3 # 30 % more biased towards exploration 
        prob = prob / np.sum(prob)

        direction = np.random.choice(4, p=prob)

        if direction == 0:
            self.yPOS -= self.step
        elif direction == 1:
            self.yPOS += self.step
        elif direction == 2:
            self.xPOS -= self.step
        elif direction == 3:
            self.xPOS += self.step

        if self.has_visited(self.xPOS, self.yPOS):
            self.energy -= 1
        else:
            self.energy += 5

        new_dist = self.intelect()
        efficiency = (old_dist - new_dist) / self.step

        if efficiency > 0.8:
            self.energy += 5

        if self.age < 500:
            self.moveEnergy = 5 * (self.age / 500)
        else:
            self.moveEnergy = 5

        self.PathMemory.append(int(self.xPOS))
        self.PathMemory.append(int(self.yPOS))
        self.xPOS, self.yPOS = np.clip(self.xPOS, 0 , worldsize), np.clip(self.yPOS, 0, worldsize)

    def intelect(self):
        dist = math.sqrt((300 - end_x) ** 2 + (300 - end_y) ** 2)
        return dist
    
def progress_bar(current, max_energy, length=20):
    progress = current/max_energy
    filled = int(round(length * progress))
    bar = '█' * filled + '-' * (length - filled)
    print(f"\rEnergy: [{bar}] {current:.1f}/{max_energy:.1f}", end="", flush=True)

def write2file():
    with open("log.json", 'w') as file:
        json.dump(creature.PathMemory, file)

def timed_operation():
    elapsed = time.time() - start_time
    Timer(1.0, timed_operation).start()

def spawn_creature():
    return FRED()

creature = FRED()

def mutate(creature, mutation_rate = 0.1, mutation_scale = 0.1): # decimal = percentage (0.1 = 10 % )
    creature.old_weights = creature.weights.copy()
    creature.old_biases = creature.biases.copy()
    creature.old_step  = creature.step
    creature.old_base_energy = creature.base_energy

    new_weights = np.random.random(creature.weights.shape) < mutation_rate
    noise_weights = np.random.normal(0, mutation_scale, creature.weights.shape)
    creature.weights += new_weights * noise_weights

    new_biases = np.random.random(creature.biases.shape) < mutation_rate
    noise_biases = np.random.normal(0, mutation_scale, creature.biases.shape)
    creature.biases += new_biases * noise_biases

    if random.random() < mutation_scale:                                    # Life expectancy mutation 
        creature.base_energy += np.random.normal(0, mutation_scale)
        creature.base_energy = max(1, creature.base_energy)

    creature.energy = creature.base_energy * creature.step

    return creature

start_x, start_y = creature.xPOS, creature.yPOS
run = True
timed_operation()   
os.system("cls")

try: 
    while run:
        time.sleep(0.1)

        if random.random() < 0.01:
            creature =  mutate(creature)

        creature.move()

        end_x, end_y = creature.xPOS, creature.yPOS

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"""
Current Pos: ({end_x:.1f}, {end_y:.1f})
Distance to Target: {FRED.intelect():.1f}
Steps Taken: {len(FRED.PathMemory)//2}
""")

        if creature.energy < 0: 
            creature.intelect()
            write2file()
            spawn_creature()

except KeyboardInterrupt:
    print(f"""
The user canceled the program. The program ran for {time.time() - start_time:.1f} seconds.
##########################################################################################
                                    Fred's Details 

Fred distance to target : {creature.intelect()}
Total movement : {end_x - start_x:.1f}, {end_y - start_y:.1f}

Fred Start : {start_x:.1f}, {start_y:.1f}
Fred Current Pos : {end_x:.1f}, {end_y:.1f}
Fred Lifetime Expectancy {creature.energy:.1f}

*/* --------- Before --------- */*
creature Weights : {creature.old_weights}
creature Biases : {creature.old_biases}
*/* -------------------------- */*

*/* --------- After --------- */*
Fred Weights : {creature.weights}
Fred Biases : {creature.biases}
*/* -------------------------- */*

##########################################################################################
""")

