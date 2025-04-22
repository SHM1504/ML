# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:42:00 2025

@author: ad

Hier ist ein einfaches Beispiel für Q-Learning im Reinforcement Learning (RL). Wir betrachten eine 2x2-Grid-Welt, in der ein Agent lernt, den optimalen Pfad zu einem Ziel zu finden.

Umwelt:
States (Zustände): 4 Zellen, nummeriert als S0, S1, S2, S3.

Actions (Aktionen): UP, DOWN, LEFT, RIGHT.

Reward (Belohnung):

Erreichen des Ziels (S3): +10

Bewegung in andere Zustände: 0

Ungültige Aktion (z. B. gegen Wand): -1
Beispiel-Grid:
S0 — S1
|     |
S2 — S3 (Ziel)



Q-Learning Algorithmus:
Q-Tabelle initialisieren (mit Nullen oder Zufallswerten).

Aktion wählen (z. B. ε-greedy: meist beste Aktion, manchmal zufällige Exploration).

Q-Wert aktualisieren:

(Formel)
α = Lernrate (z. B. 0.1),

γ
γ = Diskontfaktor (z. B. 0.9).

Beispiel-Trajektorie:
1. Start in S0:

Wähle RIGHT → erreiche S1.

Update:
    
2. In S1:

Wähle DOWN → erreiche S3 (Ziel).

Update:

3. Nächste Episode:

Wenn der Agent erneut S0 startet und RIGHT → S1 wählt:

    
Finale Q-Tabelle (nach vielen Episoden):
State	UP	DOWN	LEFT	RIGHT
S0	-1	-1	-1	0.9
S1	-1	10	-1	-1
S2	0.9	-1	-1	10
S3	0	0	0	0
Optimaler Pfad:

Von S0: RIGHT → S1, dann DOWN → S3.

Von S2: RIGHT → S3.


"""

# %%

import numpy as np

# Q-Tabelle (States x Actions)
Q = np.zeros((4, 4))  # 4 Zustände, 4 Aktionen

# Parameter
alpha = 0.1
gamma = 0.9
episodes = 1000

# Reward-Matrix (Beispiel)
R = np.array([
    [-1, -1, -1, 0],   # S0
    [-1, 10, -1, -1],   # S1
    [0.9, -1, -1, 10],  # S2
    [0, 0, 0, 0]        # S3 (Ziel)
])

# Q-Learning
for _ in range(episodes):
    s = np.random.randint(0, 4)  # Start in zufälligem Zustand
    while s != 3:  # Bis Ziel erreicht
        a = np.random.randint(0, 4)  # Zufällige Aktion (ε-greedy besser)
        s_new = a  # Vereinfachte Zustandsübergänge
        reward = R[s, a]
        Q[s, a] += alpha * (reward + gamma * np.max(Q[s_new]) - Q[s, a])
        s = s_new

print("Finale Q-Tabelle:")
print(Q)


# Hinweis: Dies ist ein stark vereinfachtes Beispiel. In der Praxis müssen Sie:

# Die Zustandsübergänge korrekt modellieren (z. B. mit einer Übergangsmatrix).

# Exploration vs. Exploitation (ε-greedy) implementieren.

# Ggf. tiefes Q-Learning (DQN) für komplexere Umgebungen verwenden.


# %%


"""
Hier ist ein konkretes, praxisnahes Beispiel für Q-Learning: Ein Roboter in einem Labyrinth, der den kürzesten Weg zu einem Ziel lernt.

Szenario: Labyrinth-Navigation
Umwelt: 5x5-Grid, Wände blockieren Bewegung.

Ziel: Roboter startet bei (0,0) und muss das Ziel bei (4,4) erreichen.

Rewards:

Ziel erreichen: +100

Bewegung in leere Zelle: -1 (ermutigt kürzeste Pfade)

Gegen Wand laufen: -5

Aktionen: UP, DOWN, LEFT, RIGHT (je nach Position ggf. ungültig).

Beispiel-Labyrinth (W = Wand, . = frei):
S . . W .
W W . W .
. . . . .
. W W W .
. . . . G


Implementierungsschritte
1. Q-Tabelle initialisieren
States: Jede Zelle (x,y) → 25 Zustände.

Aktionen: 4 Richtungen → Q-Tabelle hat Shape (25, 4).

2. Reward-Matrix definieren
3. Q-Learning Algorithmus


"""
import numpy as np
# %%    Reward-Matrix definieren

rewards = np.full((5, 5), -1)  # Default: -1 für Bewegung
rewards[4, 4] = 100            # Ziel
walls = [(0,3), (1,0), (1,1), (1,3), (3,1), (3,2), (3,3)]  # Wände
for (x,y) in walls:
    rewards[x,y] = -5           # Strafe für Wand
    
# %%    3. Q-Learning Algorithmus


# Hyperparameter
alpha = 0.1  # Lernrate
gamma = 0.9  # Diskontfaktor
epsilon = 0.2  # Exploration (ε-greedy)

# Q-Tabelle
q_table = np.zeros((25, 4))  # 25 Zustände (5x5), 4 Aktionen

# Aktionen: [UP, DOWN, LEFT, RIGHT]
actions = [(-1,0), (1,0), (0,-1), (0,1)]

for episode in range(1000):
    state = (0, 0)  # Startposition
    while state != (4, 4):  # Bis Ziel erreicht
        x, y = state
        state_idx = x * 5 + y  # Zustand als Index (0-24)
        
        # ε-greedy-Aktion wählen
        if np.random.uniform(0, 1) < epsilon:
            action_idx = np.random.randint(0, 4)  # Zufällige Aktion
        else:
            action_idx = np.argmax(q_table[state_idx])  # Beste Aktion
        
        dx, dy = actions[action_idx]
        new_x, new_y = x + dx, y + dy
        
        # Strafe für ungültige Aktionen (Wand oder Grid-Grenze)
        if new_x < 0 or new_x >= 5 or new_y < 0 or new_y >= 5 or (new_x, new_y) in walls:
            reward = -5
            new_x, new_y = x, y  # Bleibe im aktuellen Zustand
        else:
            reward = rewards[new_x, new_y]
        
        # Q-Wert aktualisieren
        new_state_idx = new_x * 5 + new_y
        q_table[state_idx, action_idx] += alpha * (
            reward + gamma * np.max(q_table[new_state_idx]) - q_table[state_idx, action_idx]
        )
        
        state = (new_x, new_y)

print("Q-Tabelle (optimale Policy):")
print(np.argmax(q_table, axis=1).reshape(5, 5))  # Zeigt beste Aktion pro Zelle
    
# %%    Ergebnisinterpretation
# Ausgabe: Eine Matrix, die die optimale Aktion pro Zelle zeigt:

# Q-Tabelle (optimale Policy):
#     [[3 3 1 0 1]
#      [0 0 1 0 1]
#      [3 3 3 3 1]
#      [0 0 0 0 1]
#      [3 3 3 3 0]]
# 0=UP, 1=DOWN, 2=RIGHT, 3=LEFT

# Beispiel: In (0,0) ist RIGHT die beste Aktion.
    
# %%    Optimaler Pfad:

# (0,0) → (0,1) → (0,2) → (1,2) → (2,2) → (2,3) → (2,4) → (3,4) → (4,4)

# %%

# Erweiterungen für die Praxis
# Tiefes Q-Learning (DQN): Ersetzt die Q-Tabelle durch ein neuronales Netz, um große Zustandsräume zu handhaben.
# Windige Umgebung: Zufällige Störungen (z. B. 20% Chance, dass Aktion fehlschlägt).
# Dynamische Hindernisse: Wände bewegen sich mit der Zeit.

