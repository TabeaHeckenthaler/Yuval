"""self written functions"""
from Setup.Maze import Maze
from Setup.Load import Load, Load_loop
# from Analysis_Functions.States import States_loop
from copy import copy
from Setup.Maze import ResizeFactors
from PhysicsEngine.Display_Pygame import Display_setup, Pygame_EventManager, Display_end, Display_renew

# from PhysicsEngine.Contact import Contact_loop2

''' Different cases: 
    (1) just play saved trajectory(don't find the contact points) ... [Display frame in for loop]
    (2) only display 1 frame (just output a .png ) ... [save a single frame at the end]
    (3) simulation (let the physics engine calculate the steps according to some forces) [o]
    (4) exp. tracking (no need to let the physics engine calculate, because we are just displaying and finding contact 
    points) 
    (5) Divide run into different attempts [in for loop check whether it is an attempt]
    (6) Find start and ending point of a trajectory [find starting and end frames in for loop]
'''

''' Functions to define for every case: step, end-condition for the loop, output '''
''' OUTPUT '''


def MainGameLoop(x, *args, interval=1, display=False, **kwargs):
    """
    Start instantiating the World and the load...
    """
    # pause = len(x.frames) < 10
    pause = False
    my_maze = Maze(*args, size=x.size, shape=x.shape, solver=x.solver)
    my_load = Load(my_maze)
    if 'moreBodies' in kwargs:
        my_maze, my_attempt_zone = kwargs['moreBodies'](my_maze, x, **kwargs)
    attempts = [False]

    if 'states' in args:
        states = []

    if 'contact' in args:
        contact = []

    lines_stat, circles_stat, points = [], [], []

    if display:
        lines_stat, circles = Display_setup(my_maze, [], [])

    """
    --- main game loop ---
    """
    running = True  # this tells us, if the simulation is still running
    i = 0  # This tells us which movie the experiment is part of, 'i' is loop counter

    # Loop that runs the simulation... 
    while running:
        """ Display the frame """
        if display:
            Display_renew(i, my_maze, *args, Trajectory=x, interval=interval, **kwargs)

        # if not(pause):
        my_load, my_maze, i = x.step(my_load, my_maze, i, pause, **kwargs)

        ''' Calculate new position for my_load  '''
        load_vertices = Load_loop(my_load)

        """ Find Contact """
        if 'contact' in args:  # if the current body is the load, search for contact points...
            contact.append(Contact_loop2(load_vertices, my_maze))
            kwargs['contact'] = contact[-1]

        ''' Any random function, that I want to inspect step wise'''
        if 'loop_function' in kwargs.keys():
            for [array, function] in kwargs['loop_function']:
                array.append(function(x, my_load, i))

        """ Display the frame """
        if display:
            running, i, pause = Pygame_EventManager(x, i,
                                                    my_maze, my_load,
                                                    points,
                                                    copy(lines_stat), circles,
                                                    *args,
                                                    pause=pause,
                                                    interval=interval,
                                                    **kwargs)

        if not pause:
            i += interval  # we start a new iteration

        if i >= len(x.frames) - 1 - interval and not pause:
            # if not (all([x_coor > finish_line for x_coor in [p[0] for p in load_vertices]])) and x.winner:
            #     x_distance = max([finish_line - x_coor for x_coor in [p[0] for p in load_vertices]])
            #     x = extend(x, 'end', x_distance, *args)
            #         '''Extended run by linear continuation in x direction until all the corners passed the ' \
            #                    'finish line '''

            # else:
            running = False  # break the loop, if we are at the end of the experimental data.
            if display:
                if len(x.frames) < 4:  # just to check the error.
                    running, i, pause = Pygame_EventManager(x, i,
                                                            my_maze, my_load,
                                                            load_vertices, points,
                                                            copy(lines_stat), circles,
                                                            attempts, pause=True)

                running = Display_end('../OldFunctions/ErrorFinding.png')

    """ Return the right variables """
    if 'loop_function' in kwargs.keys():
        return x, [array for array, function in kwargs['loop_function']]

    if 'contact' in args:
        return x, contact

    if 'attempt' in args:
        return x, attempts

    elif 'states' in args:
        return x, states

    else:
        return x
