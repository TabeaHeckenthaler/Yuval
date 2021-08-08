
from trajectory import Get
from Classes_Experiment.humans import Humans
from Classes_Experiment.forces import participants_force_arrows

''' Display a experiment '''
# names are found in P:\Tabea\PyCharm_Data\AntsShapes\Pickled_Trajectories\Human_Trajectories
solver = 'human'
x = Get('medium_20201221135753_20201221140218', solver)
x.participants = Humans(x)
x.play(forces=[participants_force_arrows])

# press Esc to stop the display
