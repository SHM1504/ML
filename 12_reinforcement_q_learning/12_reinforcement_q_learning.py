# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 10:16:21 2025

@author: ad

 *** Reinforcement Q-Learning from Scratch in Python with OpenAI Gym ***
Quelle: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ 

Reinforcement Learning (Bestärkendes Lernen) Analogie

- ein Hund ist ein „Agent“ in einer 'environment'
- Agenten reagieren, indem sie eine Aktion ausführen, um von einem „Zustand (state)“
 in einen anderen „Zustand“ überzugehen
- Nach dem Übergang kann er eine Belohnung oder eine Strafe erhalten. 
- Die Strategie ist die Wahl einer Handlung in einem bestimmten Zustand in der
  Erwartung eines besseren Ergebnisses.
- Ziel: ... die maximalen Belohnungen über das gesamte Training hinweg zu optimieren 
- Der Belohnungsagent hängt nicht nur vom aktuellen Zustand ab, sondern von der
  gesamten Geschichte der Zustände
  
Der Prozess des Verstärkungslernens besteht aus diesen einfachen Schritten:

- Beobachtung der Umgebung
- Entscheiden, wie man sich mit einer bestimmten Strategie verhalten soll
- Entsprechend handeln
- Erhalt einer Belohnung oder Bestrafung
- Lernen aus den Erfahrungen und Verfeinerung der Strategie
- Iterieren, bis eine optimale Strategie gefunden ist  

Example Design: Self-Driving Cab
 * Die Aufgabe des Smartcabs ist es, den Fahrgast an einem Ort abzuholen und
   an einem anderen abzusetzen.
   
   ... modeling an RL solution to this problem: rewards, states, and actions.
   
1. Rewards (Belohnung)
   - eine hohe Belohnung für ein erfolgreiches Absetzen   
   - Der Agent sollte bestraft(penalized) werden, wenn er versucht, einen Fahrgast
     an falschen Orten abzusetzen 
   - eine leicht negative Belohnung erhalten, wenn er es nicht nach jedem
     Zeitschritt zum Zielort schafft  

2. State Space (Zustandsraum)
    - Beim Reinforcement Learning trifft der Agent auf einen Zustand und ergreift
    dann Maßnahmen entsprechend dem Zustand, in dem er sich befindet.

    - Der Zustandsraum ist die Menge aller möglichen Situationen, die unser Taxi
    durchlaufen könnte. Der Zustand sollte nützliche Informationen enthalten,
    die der Agent benötigt, um die richtige Aktion auszuführen.
    
    - ... Parkplatz in ein 5x5-Gitter einteilen, 4 mögliche Ziele (R,G,B,Y), und 
    fünf (4 + 1) Fahrgaststandorte.
    
    taxi environment hat 5 x 5 x 5 x 4 = 500 total possible states.
    
3. Action Space
    - Die Aktion kann darin bestehen, sich in eine bestimmte Richtung zu bewegen
      oder einen Fahrgast aufzunehmen/abzusetzen.

    - Somit, haben wir sechs mögliche Aktionen:
      Süden, Norden, Osten, Westen, Abholung, Absetzen
      
    - -1 (penalty) für jede getroffene Wand geben und das Taxi wird sich 
    nirgendwo hinbewegen
     
    
Neues Environment:
conda create -n rl gymnasium=0.26 spyder matplotlib numpy tqdm
    
"""
# %%    Gym's interface

import gymnasium as gym

from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as img
from IPython.display import clear_output
import numpy as np
import random

# %%

im = img.imread('Reinforcement_Learning_Taxi_Env.width-1200.png')
plt.figure(dpi=1200)
plt.axis('off')
plt.imshow(im)
plt.show()

#%%

# The core gym interface is 'env', 
env = gym.make("Taxi-v3", render_mode='ansi').env
# %%

# Setzt die Umgebung zurück und gibt einen zufälligen Ausgangszustand zurück.
env.reset()

# Rendert einen Frame der Umgebung (hilfreich für die Visualisierung der Umgebung)
# strip() und var frame ist für bessere Darstellung in der Konsole Umgebung
frame = env.render().strip()
print(frame)


# %%    Erinnerung an das Problem

# Es gibt 4 Orte (mit verschiedenen Buchstaben gekennzeichnet), und unsere Aufgabe
# ist es, den Fahrgast an einem Ort abzuholen und ihn an einem anderen abzusetzen.
# Wir erhalten +20 Punkte für das erfolgreiche Absetzen und verlieren 1 Punkt für
# jeden Zeitschritt, den wir brauchen. Außerdem gibt es 10 Punkte Strafe für
# illegale Abhol- und Absetzaktionen.


print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

# R, G, Y, B sind die möglichen Abhol- und Zielorte. Der blaue Buchstabe steht
# für den aktuellen Abholort des Fahrgastes und der violette Buchstabe für den
# aktuellen Zielort.

# wir haben einen Aktionsraum der Größe 6 und einen Zustandsraum der Größe 500
# was wir brauchen, ist eine Möglichkeit, einen Zustand eindeutig zu identifizieren,
# indem wir jedem möglichen Zustand eine eindeutige Nummer zuweisen, und RL
# lernt, eine Aktionsnummer von 0-5 zu wählen, wobei:

# 0 = south
# 1 = north
# 2 = east
# 3 = west
# 4 = pickup
# 5 = dropoff

# Die 500 Zustände entsprechen einer Kodierung des Standorts des Taxis, des
# Standorts des Fahrgastes und des Zielorts.    

# -----------------------------------------------
# Reinforcement Learning lernt eine Abbildung von Zuständen auf die optimale
# Aktion, die in diesem Zustand auszuführen ist, durch Exploration, d.h.
# der Agent erkundet die Umgebung und führt Aktionen auf der Grundlage von
# in der Umgebung definierten Belohnungen aus.

# Die optimale Aktion für jeden Zustand ist die Aktion mit der höchsten
# kumulativen langfristigen Belohnung.
# ----------------------------------------------

# %%    Back to our illustration
# Wir erinnern uns, dass sich das Taxi in Zeile 3, Spalte 1 befindet, unser 
# Fahrgast an Position 2 und unser Ziel an Position 0. Mit der Taxi-v3-
# Zustandskodierungsmethode können wir Folgendes tun

state = env.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
print("State:", state)

env.s = state

frame = env.render().strip()
print(frame)

# State: 328
# den Zustand unserer Illustration ergibt 328
# +---------+
# |R: | : :G|
# | : : : : |
# | : : : : |
# | | : | : |
# |Y| : |B: |
# +---------+

# %%    The Reward Table {states x actions} matrix
# Wenn die Taxi-Umgebung erstellt wird, wird auch eine anfängliche Belohnungstabelle
# erstellt, die „P“ genannt wird.
# State: 328

env.P[328]

# {action: [(probability, nextstate, reward, done)]}.

# {0: [(1.0, 428, -1, False)],
#  1: [(1.0, 228, -1, False)],
#  2: [(1.0, 348, -1, False)],
#  3: [(1.0, 328, -1, False)],
#  4: [(1.0, 328, -10, False)],
#  5: [(1.0, 328, -10, False)]}




# %%
# Dann können wir den Zustand der Umgebung manuell mit env.env.s unter Verwendung
# dieser kodierten Zahl einstellen. Sie können mit den Zahlen herumspielen und
# sehen, wie sich das Taxi, der Fahrgast und das Ziel bewegen.

env.s = 328  # set environment to illustration's state

epochs = 0
penalties, reward = 0, 0

frames = [] # for animation

done = False

# env.step(action) Step the environment by one timestep. 
# Returns:
# ------------    
# observation: Beobachtungen der Umgebung
# reward: Ob die Aktion nützlich war oder nicht
# done: Zeigt an, ob wir einen Passagier erfolgreich aufgenommen und abgesetzt 
#       haben, auch eine Episode genannt
# info: Zusätzliche Informationen wie Leistung und Latenz für Debugging-Zwecke

# Note: We are using the .env on the end of make to avoid training stopping 
# at 200 iterations, which is the default for the new version of Gym

# env.action_space.sample(): Methode wählt automatisch eine zufällige Aktion
# aus der Menge aller möglichen Aktionen aus.

while not done:
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)

    if reward == -10:
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )

    epochs += 1
    
    
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))

#%%

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)


# %%
"""        Q-learning   """


# Im Wesentlichen ermöglicht das Q-Learning dem Agenten, die Belohnungen der
# Umgebung zu nutzen, um im Laufe der Zeit zu lernen, welche Aktion in einem
# bestimmten Zustand am besten zu wählen ist.  

# Die in der Q-Tabelle gespeicherten Werte werden als Q-Werte bezeichnet und
# entsprechen einer Kombination (Zustand, Aktion).

# Ein Q-Wert für eine bestimmte Zustands-Aktions-Kombination ist repräsentativ
# für die „Qualität“ einer Aktion, die in diesem Zustand ausgeführt wird.
# Bessere Q-Werte bedeuten bessere Chancen auf größere Belohnungen.

###         Q-Tabelle     ###
# Die Q-Tabelle ist eine Matrix mit einer Zeile für jeden Zustand (500) und
# einer Spalte für jede Aktion (6). Sie wird zunächst mit 0 initialisiert,
# und die Werte werden nach dem Training aktualisiert

# %%

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# %%

"""Training the agent"""


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

# Im ersten Teil von while not done entscheiden wir, ob wir eine zufällige
# Aktion wählen oder die bereits berechneten Q-Werte nutzen wollen. Dazu
# verwenden wir einfach den Epsilon-Wert und vergleichen ihn mit der Funktion
# random.uniform(0, 1), die eine beliebige Zahl zwischen 0 und 1 zurückgibt.

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = [env.action_space.sample()] # Explore action space
        else:
            
            action = [np.argmax(q_table[state[0]])] # Exploit learned values

# Wir führen die gewählte Aktion in der Umgebung aus, um den next_state
# und reward für die Durchführung der Aktion zu erhalten. Danach berechnen
# wir den maximalen Q-Wert (np.max(q-Wert)) für die Aktionen, die dem 
# next_state entsprechen,
# und damit können wir unseren Q-Wert einfach auf den new_q_Wert aktualisieren:

        next_state, reward, done, truncated, info = env.step(action[0]) 
        
        old_value = q_table[state[0], action[0]]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state[0], action[0]] = new_value

        if reward == -10:
            penalties += 1

        state = (next_state, state[1:])
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")

# %%

q_table[328]

# %%    Evaluating the agent

"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100
frames = []

for _ in range(episodes):
    state = env.reset()[0]
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

# %%

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)