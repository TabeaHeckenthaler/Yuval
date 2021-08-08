# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:24:09 2020

@author: tabea
"""
from scipy.spatial import cKDTree
import numpy as np
import glob
from os import (listdir, getcwd, mkdir, path)
import pickle
import shutil
from Setup.MazeFunctions import BoxIt
from PhysicsEngine import Box2D_GameLoops

""" Making Directory Structure """
shapes = {'ant': ['SPT', 'H', 'I', 'T', 'RASH', 'LASH'],
          'human': ['SPT']}
sizes = {'ant': ['XS', 'S', 'M', 'L', 'SL', 'XL'],
         'human': ['S', 'M', 'L'],
         'humanhand': ''}
solvers = ['ant', 'human', 'humanhand', 'dstar']

home = 'C:\\Users\\tabea\\PycharmProjects\\AntsShapes\\'
data_home = '{sep}{sep}phys-guru-cs{sep}ants{sep}Tabea{sep}PyCharm_Data{sep}AntsShapes{sep}'.format(sep=path.sep)
work_dir = data_home + 'Pickled_Trajectories\\'
AntSaverDirectory = work_dir + 'Ant_Trajectories'
HumanSaverDirectory = work_dir + 'Human_Trajectories'
HumanHandSaverDirectory = work_dir + 'HumanHand_Trajectories'
DstarSaverDirectory = work_dir + 'Dstar_Trajectories'
SaverDirectories = {'ant': AntSaverDirectory,
                    'human': HumanSaverDirectory,
                    'humanhand': HumanHandSaverDirectory,
                    'dstar': DstarSaverDirectory}

length_unit = {'ant': 'cm',
               'human': 'm',
               'humanhand': 'cm',
               'dstar': 'cm'}


def length_unit_func(solver):
    return length_unit[solver]


def communication(filename, solver):
    if solver != 'human':
        return False
    else:
        from Classes_Experiment.humans import excel_worksheet_index, get_sheet
        index = excel_worksheet_index(filename)
        return get_sheet().cell(row=index, column=5).value == 'C'


def maze_size(size):
    maze_s = {'Large': 'L',
              'Medium': 'M',
              'Small Far': 'S',
              'Small Near': 'S'}
    if size in maze_s.keys():
        return maze_s[size]
    else:
        return size


def time(x, condition):
    if condition == 'winner':
        return x.time
    if condition == 'all':
        if x.winner:
            return x.time
        else:
            return 60 * 40  # time in seconds after which the maze supposedly would have been solved?


def Directories():
    if not (path.isdir(AntSaverDirectory)):
        breakpoint()
        mkdir(AntSaverDirectory)
        mkdir(AntSaverDirectory + path.sep + 'OnceConnected')
        mkdir(AntSaverDirectory + path.sep + 'Free_Motion')
        mkdir(AntSaverDirectory + path.sep + 'Free_Motion' + path.sep + 'OnceConnected')
    if not (path.isdir(HumanSaverDirectory)):
        mkdir(HumanSaverDirectory)
    if not (path.isdir(HumanHandSaverDirectory)):
        mkdir(HumanHandSaverDirectory)
    if not (path.isdir(DstarSaverDirectory)):
        mkdir(DstarSaverDirectory)
    return


def NewFileName(old_filename, size, shape, expORsim):
    if expORsim == 'sim':
        counter = int(len(glob.glob(size + '_' + shape + '*_' + expORsim + '_*')) / 2 + 1)
        # findall(r'[\d.]+', 'TXL1_sim_255')[1] #this is a function able to read the last digit of the string
        filename = size + '_' + shape + '_sim_' + str(counter)
    if expORsim == 'exp':
        filename = old_filename.replace('.mat', '')
        if shape.endswith('ASH'):
            filename = filename.replace(old_filename.split('_')[0], size + '_' + shape)
        else:
            filename = filename.replace(size + shape, size + '_' + shape)
    return filename


def Get(filename, solver, address=None):
    if address is None:
        if path.isfile(SaverDirectories[solver] + path.sep + filename):
            address = SaverDirectories[solver] + path.sep + filename

        elif path.isfile(SaverDirectories[solver] + path.sep + 'OnceConnected' + path.sep + filename):
            print('This must be an old file.... ')
            address = SaverDirectories[solver] + path.sep + 'OnceConnected' + path.sep + filename

        elif path.isfile(SaverDirectories[solver] + path.sep + 'Free_Motion' + path.sep + filename):
            address = SaverDirectories[solver] + path.sep + 'Free_Motion' + path.sep + filename

        elif path.isfile(
                SaverDirectories[solver] + path.sep + 'Free_Motion' + path.sep + 'OnceConnected' + path.sep + filename):
            print('This must be an old file.... ')
            address = SaverDirectories[
                          solver] + path.sep + 'Free_Motion' + path.sep + 'OnceConnected' + path.sep + filename
        else:
            print('I cannot find this file: ' + filename)
            return Trajectory()

    with open(address, 'rb') as f:
        x = pickle.load(f)
    if type(x.participants) == list:
        delattr(x, 'participants')
        Save(x)
    return x


def Save(x, address=None):
    Directories()
    if address is None:
        if x.solver in solvers:
            if x.free:
                address = SaverDirectories[x.solver] + path.sep + 'Free_Motion' + path.sep + x.filename
            else:
                address = SaverDirectories[x.solver] + path.sep + x.filename
        else:
            address = getcwd()

    with open(address, 'wb') as f:
        try:
            pickle.dump(x, f)
            print('Saving ' + x.filename + ' in ' + address)
        except pickle.PicklingError as e:
            print(e)
    # move_tail(x)
    return


def move_tail(x):
    if not x.free:
        origin_directory = SaverDirectories[x.solver]
        goal_directory = SaverDirectories[x.solver] + path.sep + 'OnceConnected'
    else:
        origin_directory = SaverDirectories[x.solver] + path.sep + 'Free_Motion'
        goal_directory = SaverDirectories[x.solver] + path.sep + 'Free_Motion' + path.sep + 'OnceConnected'

    for tailFiles in x.VideoChain[1:]:
        if path.isfile(path.join(origin_directory, tailFiles)):
            shutil.move(path.join(origin_directory, tailFiles), goal_directory)


trackedAntMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Aviram{4}Shapes Results'.format(path.sep, path.sep, path.sep,
                                                                                        path.sep, path.sep)
trackedHumanMovieDirectory = '{0}{1}phys-guru-cs{2}ants{3}Tabea{4}Human Experiments{5}Output'.format(path.sep, path.sep,
                                                                                                     path.sep, path.sep,
                                                                                                     path.sep, path.sep)
trackedHumanHandMovieDirectory = 'C:\\Users\\tabea\\PycharmProjects\\ImageAnalysis\\Results\\Data'


def MatlabFolder(solver, size, shape, free):
    if solver == 'ant':
        shape_folder_naming = {'LASH': 'Asymmetric H', 'RASH': 'Asymmetric H', 'ASH': 'Asymmetric H',
                               'H': 'H', 'I': 'I', 'LongT': 'Long T',
                               'SPT': 'Special T', 'T': 'T'}
        if not free:
            return trackedAntMovieDirectory + path.sep + 'Slitted' + path.sep + shape_folder_naming[
                shape] + path.sep + size + path.sep + 'Output Data'
        if free:
            return trackedAntMovieDirectory + path.sep + 'Free' + path.sep + 'Output Data' + path.sep + \
                   shape_folder_naming[shape]
    if solver == 'human':
        if not free:
            return trackedHumanMovieDirectory + path.sep + size + path.sep + 'Data'
        if free:
            return trackedHumanMovieDirectory + path.sep + size + path.sep + 'Data'
    if solver == 'humanhand':
        return trackedHumanHandMovieDirectory

    else:
        print('MatlabFolder: who is solver?')


class Trajectory:
    def __init__(self,
                 size=None, shape=None, solver=None,
                 filename=None,
                 free=False, fps=50,
                 winner=bool,
                 x_error=None, y_error=None, angle_error=None, falseTracking=None,
                 **kwargs):
        self.shape = shape  # shape (maybe this will become name of the maze...)
        self.size = size  # size
        self.solver = solver

        if 'old_filename' in kwargs:
            self.filename = NewFileName(kwargs['old_filename'], self.size, self.shape, 'exp')
        else:
            self.filename = filename  # filename: shape, size, path length, sim/ants, counter
        self.free = free
        self.VideoChain = [self.filename]
        self.fps = fps  # frames per second

        self.position = np.empty((1, 2), float)  # np.array of x and y positions of the centroid of the shape
        self.angle = np.empty((1, 1), float)  # np.array of angles while the shape is moving
        self.frames = np.empty(0, float)
        self.tracked_frames = []

        if x_error is None:
            x_error = [0]
        if y_error is None:
            y_error = [0]
        if angle_error is None:
            angle_error = [0]
        if falseTracking is None:
            falseTracking = [[]]
        self.x_error, self.y_error, self.angle_error = x_error, y_error, angle_error
        self.falseTracking = falseTracking
        self.winner = winner  # whether the shape crossed the exit
        self.state = np.empty((1, 1), int)

    def __bool__(self):
        return self.winner

    def __str__(self):
        string = '\n' + self.filename
        return string

    def old_filenames(self, i):
        if i >= len(self.VideoChain):
            return 'No video found (maybe because I extended)'

        if self.shape[1:] == 'ASH':
            if self.VideoChain[i].split('_')[0] + '_' + \
                    self.VideoChain[i].split('_')[1] == self.size + '_' + self.shape:
                old = self.VideoChain[i].replace(
                    self.VideoChain[i].split('_')[0] + '_' + self.VideoChain[i].split('_')[1],
                    self.size + self.shape[1:]) \
                      + '.mat'
            else:
                print('Something strange in x.old_filenames of x = ' + self.filename)
            #     # this is specifically for 'LASH_4160019_LargeLH_1_ants (part 1).mat'...
            #     old = self.VideoChain[i] + '.mat'
        else:
            old = self.VideoChain[i].replace(self.size + '_' + self.shape, self.size + self.shape) + '.mat'
        return old

    # Find the size and shape from the filename
    def shape_and_size(self, old_filename):
        if self.size == str('') and self.solver != 'humanhand':
            if len(old_filename.split('_')[0]) == 2:
                self.size = old_filename.split('_')[0][0:1]
                self.shape = old_filename.split('_')[0][1]
            if len(old_filename.split('_')[0]) == 3:
                self.size = old_filename.split('_')[0][0:2]
                self.shape = old_filename.split('_')[0][2]
            if len(old_filename.split('_')[0]) == 4:
                self.size = old_filename.split('_')[0][0:1]
                self.shape = old_filename.split('_')[0][1:4]  # currently this is only for size L and shape SPT
            if len(old_filename.split('_')[0]) == 5:
                self.size = old_filename.split('_')[0][0:2]
                self.shape = old_filename.split('_')[0][2:5]
        # now we figure out, what the zone is, if the arena size were equally scaled as the load and exit size.
        # arena_length, arena_height, x.exit_size, wallthick, slits, resize_factor = getMazeDim(x.shape, x.size)

    def participants(self):
        from Classes_Experiment.ants import Ants
        from Classes_Experiment.humans import Humans
        from Classes_Experiment.mr_dstar import Mr_dstar
        from Classes_Experiment.humanhand import Humanhand

        dc = {'ant': Ants,
              'human': Humans,
              'humanhand': Humanhand,
              'dstar': Mr_dstar
              }
        return dc[self.solver](self)

    def timer(self):
        return (len(self.frames) - 1) / self.fps

    def ZonedAngle(self, index, my_maze, angle_passed):
        angle_zoned = angle_passed
        if not (self.InsideZone(index, my_maze)) and self.position[-1][0] < self.zone[2][0]:
            angle_zoned = self.angle_zoned[index]
        return angle_zoned

    def ZonedPosition(self, index, my_maze):
        # first we check whether load is inside zone, then we check, whether the load has passed the first slit
        if self.position.size < 3:
            if not (self.InsideZone(index, my_maze)) and self.position[0] < self.zone[2][0]:
                zone = cKDTree(BoxIt(self.zone, 0.01))  # to read the data, type zone.data
                position_zoned = zone.data[zone.query(self.position)[1]]
            else:
                position_zoned = self.position
            if position_zoned[1] > 14.9:
                breakpoint()
        else:
            if not (self.InsideZone(index, my_maze)) and self.position[index][0] < self.zone[2][0]:
                zone = cKDTree(BoxIt(self.zone, 0.01))  # to read the data, type zone.data
                position_zoned = zone.data[zone.query(self.position[index])[1]]
            else:
                position_zoned = self.position[index]
            if position_zoned[1] > 14.9:
                breakpoint()
        return position_zoned

    def InsideZone(self, index, Maze):
        if self.position.size < 3:
            x = (Maze.zone[2][0] > self.position[0] > Maze.zone[0][0])
            y = (Maze.zone[1][1] > self.position[1] > Maze.zone[0][1])
        else:
            x = (Maze.zone[2][0] > self.position[index][0] > Maze.zone[0][0])
            y = (Maze.zone[1][1] > self.position[index][1] > Maze.zone[0][1])
        return x and y

    def first_nonZero_State(self):
        for i in range(1, len(self.state[1:] - 1)):
            if self.state[i] != 0:
                return i
        print('I did not find a non-Zero state')

    def first_Contact(self):
        contact = self.play(1, 'contact')[1]
        for i in range(1, len(self.state[1:] - 1)):
            if contact[i].size != 0 or self.state[i] != 0:
                return i

        print('I did not find a Contact state or a non-Zero state')

    def step(self, my_load, my_maze, i, pause, **kwargs):
        my_load.position.x, my_load.position.y, my_load.angle = self.position[i][0], self.position[i][1], self.angle[i]
        return my_load, my_maze, i

    def play(self, *args, interval=1, **kwargs):
        from copy import deepcopy
        x = deepcopy(self)

        if hasattr(x, 'contact'):
            delattr(x, 'contact')

        if x.frames.size == 0:
            x.frames = np.array([fr for fr in range(x.angle.size)])

        if 'indices' in kwargs.keys():
            f1, f2 = int(kwargs['indices'][0]), int(kwargs['indices'][1]) + 1
            x.position, x.angle = x.position[f1:f2, :], x.angle[f1:f2]
            x.frames = x.frames[int(f1):int(f2)]

        if 'L_I_425' in x.filename:
            args = args + ('L_I1',)

        return Box2D_GameLoops.MainGameLoop(x, *args, display=True, interval=interval, **kwargs)
