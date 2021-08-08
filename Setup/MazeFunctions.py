# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:41:04 2020

@author: tabea
"""

## Give CreateSlitObject the world object, and the x position of the left corner of the slit (L), the maze height (H), the exit size (e) and the wall thickness (d). 
''' General Functions '''
from Box2D import (b2ContactListener)
import numpy as np
import pygame
from matplotlib import pyplot as plt


# from os import listdir, path, getcwd

def threads_over_lists(fn):
    def wrapped(x):
        if isinstance(x, list):
            return [fn(e) for e in x]
        return fn(x)

    return wrapped


def FrametoIndex(x, frame, **kwargs):
    if 'minIndex' in kwargs:
        return [e for e in np.where(x.frames == frame)[0] if e > kwargs['minIndex']]
    return list(np.where(x.frames == frame)[0])


def IsExperiment(Exp):
    return not (Exp.startswith('ErrorPickle')) and not (Exp.startswith('Stuff')) and not (
        Exp.startswith('desktop')) and not (Exp.startswith('Free')) and not (Exp.startswith('Once')) and not (
        Exp.startswith('Traj')) and not Exp == 'AssymetricH_Series' and not Exp.startswith('DontLoad')


def RotateAndShiftSystem(position, offset, rotate):
    position[:, 0] = position[:, 0] - offset[0]
    position[:, 1] = position[:, 1] - offset[1]

    if rotate:
        hyp = np.sqrt(((position[0, 0]) ** 2 + (
            position[0, 1]) ** 2))  # this is the length of the direct route from beginning to end
        alpha = - np.arccos(abs(position[0, 0]) / hyp)  # angle at which we want to rotate the system
        if position[1, 0] - position[-1, 0] < 0 and position[0, 1] < 0:
            alpha = alpha - np.pi
        if position[1, 0] - position[-1, 0] < 0 < position[0, 1]:
            alpha = np.pi - alpha
    else:
        alpha = 0
    x_new = position[:, 0] * np.cos(alpha) - position[:, 1] * np.sin(alpha)
    y_new = position[:, 0] * np.sin(alpha) + position[:, 1] * np.cos(alpha)
    position_new = np.transpose(np.vstack((x_new, y_new)))

    # position_new[:,0] = position_new[:,0] - offset[0]
    # position_new[:,1] = position_new[:,1] - offset[1]

    if abs(position_new[0, 1]) > 0.1:
        plt.plot(position_new[:, 0], position_new[:, 1]);
        plt.show()
        breakpoint()
    return position_new


def PlotPolygon(body, arena_height, color, *vargs):
    # Go through all the fixtures in the body (this is important for the H shape, where there are more than one
    # fixtures for example)
    for fixture in body.fixtures:
        vertices = [(body.transform * v) for v in fixture.shape.vertices]
        v = np.array(vertices)
        plt.plot(np.append(v[:, 0], v[0, 0]),
                 np.append((v[:, 1]), (v[0, 1])),
                 # color + '-', markersize=10)   
                 color=color, markersize=10)
        if 'fill' in vargs:
            pass
            # I think matplotlib.patches woudl be useful here


def MeasureDistance(position1, position2, angle1, angle2, averRad, rot=True, **kwargs):  # returns distance in cm.
    archlength = 0
    if position1.ndim == 1:  # For comparing only two positions
        translation = np.linalg.norm(position1[:2] - position2[:2])
        if rot:
            archlength = abs(angle1 - angle2) * averRad

    else:  # For comparing more than 2 positions
        # translation = np.sqrt(np.sum(np.power((position1[:, :2] - position2[:, :2]), 2), axis=1))
        translation = np.linalg.norm(position1[:, :2] - position2[:, :2])
        if rot:
            archlength = abs(angle1[:] - angle2[:]) * averRad
    return translation + archlength


def ClosestCorner(vertices, gravCenter):
    corners = np.array([vertices[0].x, vertices[0].y])
    for i in range(1, len(vertices)):
        corners = np.vstack((corners, np.array([vertices[i].x, vertices[i].y])))

    Distances = np.sqrt(np.power((corners - gravCenter), 2).sum(axis=1))
    closestCorner = corners[np.argmin(Distances), :]
    return closestCorner


def DrawGrid(window, width, height, PPM, SCREEN_HEIGHT):
    from PhysicsEngine.Display_Pygame import colors
    block = 2
    block_size = block * PPM
    # for y in range(round(height / 2)):
    #     for x in range(round(width / 2)):
    for y in range(np.int(np.ceil(height / block)+1)):
        for x in range(np.int(np.ceil(width / block))):
            rect = pygame.Rect(x * block_size, SCREEN_HEIGHT -
                               y * block_size, block_size, block_size)
            pygame.draw.rect(window, colors['grid'], rect, 1)


class myContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)

    def BeginContact(self, contact):
        pass

    def EndContact(self, contact):
        pass

    def PreSolve(self, contact, oldManifold):
        pass

    def PostSolve(self, contact, impulse):
        pass


def BoxIt(corners, stepSize, **kwargs):  # corners go from left bottom, to left top, to right top, to right bottom
    corners = np.array(corners)
    TwoBottom = corners[corners[:, 1].argsort()][0:2]
    TwoTop = corners[corners[:, 1].argsort()][2:4]

    corners = np.vstack([TwoBottom[TwoBottom[:, 0].argsort()][0],
                         TwoTop[TwoTop[:, 0].argsort()][0],
                         TwoTop[TwoTop[:, 0].argsort()][1],
                         TwoBottom[TwoBottom[:, 0].argsort()][1]])

    y_Num = int(abs(corners[0][1] - corners[1][1]) / stepSize)
    x_Num = int(abs(corners[2][0] - corners[0][0]) / stepSize)

    bottom = np.transpose(np.vstack((np.linspace(corners[0][0],
                                                 corners[3][0],
                                                 num=x_Num),
                                     np.linspace(corners[0][1],
                                                 corners[3][1],
                                                 num=x_Num))))

    upper = np.transpose(np.vstack((np.linspace(corners[1][0],
                                                corners[2][0],
                                                num=x_Num),
                                    np.linspace(corners[1][1],
                                                corners[2][1],
                                                num=x_Num),)))

    left = np.transpose(np.vstack((np.linspace(corners[0][0],
                                               corners[1][0],
                                               num=y_Num),
                                   np.linspace(corners[0][1],
                                               corners[1][1],
                                               num=y_Num))))

    right = np.transpose(np.vstack((np.linspace(corners[3][0],
                                                corners[2][0],
                                                num=y_Num),
                                    np.linspace(corners[3][1],
                                                corners[2][1],
                                                num=y_Num))))

    if kwargs.get('without') is not None:
        kwargs['without'] = np.array([0, 0])
        return np.vstack((bottom, upper, left))

    return np.vstack((bottom, upper, right, left))


def ConnectAngle(angle, shape):
    # This function writes the angle as absolute angle compared to the beginning angle (which was within a certain 0
    # to 2pi range) Write the first angle in the connected angle array
    from Setup.Load import periodicity
    ''' turn the shape by 2/p * np.pi and it looks the same'''
    p = periodicity[shape]

    ''' Wrap it! '''
    angle = (angle + np.pi / p) % (2 * np.pi / p) - np.pi / p

    ''' Make NaNs and the adjacent 5 values NaNs'''
    original_nan = np.where(np.isnan(angle))[0]

    for i in range(5):
        if len(original_nan) > 0 and original_nan[-1] - i > 0:
            angle[original_nan - i] = np.NaN
        elif len(original_nan) > 0 and len(angle) - 1 > original_nan[-1] + i:
            angle[original_nan + i] = np.NaN
    not_new_nan = ~np.isnan(angle)

    ''' get rid of all NaNs '''
    angle = angle[~np.isnan(angle)]

    # ok = ~np.isnan(angle)
    # xp = ok.ravel().nonzero()[0]
    # fp = angle[~np.isnan(angle)]
    # x  = np.isnan(angle).ravel().nonzero()[0]

    ''' unwrap '''
    # angle[np.isnan(angle)] = np.interp(x, xp, fp)
    unwraped = 1 / p * np.unwrap(p * angle)
    returner = np.empty([len(not_new_nan)])

    ''' reinsert NaNs '''
    i, ii = 0, 0
    for insert in not_new_nan:
        if insert:
            returner[ii] = unwraped[i]
            i = i + 1
        else:
            returner[ii] = np.NaN
        ii = ii + 1
    # unwraped[originalNaN[0]] = np.NaN

    return returner

