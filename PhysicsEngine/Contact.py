from scipy.spatial import cKDTree
from Setup.MazeFunctions import BoxIt
import numpy as np
from Setup.Maze import Maze, maze_corners
from Setup.Load import Load_loop

distance_upper_bound = 0.04


# maximum distance between fixtures to have a contact (in cm)


def theta(r):
    [x, y] = r
    if x > 0:
        return np.arctan(y / x)
    elif x < 0:
        return np.arctan(y / x) + np.pi
    elif x == 0 and y != 0:
        return np.sign(y) * np.pi / 2


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def Contact_loop2(load, maze):
    # this function takes a list of corners (lists have to list rectangles in sets of 4 corners)
    # and checks, whether a rectangle from the load overlaps with a rectangle from the

    # first check, whether we are in the middle of the maze, where there is no need for heavy calculations
    # approximate_extent = np.max([2 * np.max(np.abs(np.array(list(fix.shape)))) for fix in load.fixtures])
    #
    # if approximate_extent + 0.1 < load.position.x < min(maze.slits) - approximate_extent - 0.1 and \
    #         approximate_extent + 0.1 < load.position.y < maze.arena_height - approximate_extent - 0.1:
    #     return False
    #
    # elif max(maze.slits) + approximate_extent + 0.1 < load.position.x:
    #     return False

    # if we are close enough to a boundary then we have to calculate all the vertices.
    load_corners = Load_loop(load)
    maze_corners1 = maze_corners(maze)
    for load_NumFixture in range(int(len(load_corners) / 4)):
        load_vertices_list = load_corners[load_NumFixture * 4:(load_NumFixture + 1) * 4] \
                             + [load_corners[load_NumFixture * 4]]

        for maze_NumFixture in range(int(len(maze_corners1) / 4)):
            maze_vertices_list = maze_corners1[maze_NumFixture * 4:(maze_NumFixture + 1) * 4] \
                                 + [maze_corners1[maze_NumFixture * 4]]
            for i in range(4):
                for ii in range(4):
                    if intersect(load_vertices_list[i], load_vertices_list[i + 1],
                                 maze_vertices_list[ii], maze_vertices_list[ii + 1]):
                        return True

    return np.any([f.TestPoint(load.position) for f in maze.body.fixtures])


def Contact_loop(load_vertices, my_maze):
    contact = []
    edge_points = []
    for NumFixture in range(int(len(load_vertices) / 4)):
        edge_points = edge_points + BoxIt(load_vertices[NumFixture * 4:(NumFixture + 1) * 4],
                                          distance_upper_bound).tolist()
    load_tree = cKDTree(edge_points)

    in_contact = load_tree.query(my_maze.slitTree.data, distance_upper_bound=distance_upper_bound)[1] < \
                 load_tree.data.shape[0]

    if np.any(in_contact):
        contact = contact + my_maze.slitTree.data[np.where(in_contact)].tolist()
    return contact


def find_Impact_Points(x, my_maze, *args, **kwargs):
    x, contact = x.play(1, 'contact', *args, *kwargs)
    wall_contacts = np.where([len(con) > 0
                              # x component close to the exit wall
                              and con[0][0] > my_maze.slits[0] - 1
                              #  and (abs(con[0][1] - my_maze.arena_height / 2 - my_maze.exit_size / 2) < 2
                              #       or abs(con[0][1] - my_maze.arena_height / 2 + my_maze.exit_size / 2) < 2)
                              for con in contact])[0]

    # only if its not a to short contact!
    wall_contacts = [c for i, c in enumerate(wall_contacts) if abs(c - wall_contacts[i - 1]) < 2]

    impact_indices = list(wall_contacts[0:1]) + [c for i, c in enumerate(wall_contacts)
                                                 if c - wall_contacts[i - 1] > int(x.fps * 2)]
    return impact_indices, contact


def reduce_contact_points(contact):
    # only the contact points that are far enough away from each other.
    if len(contact) == 0:
        return []
    else:
        a = np.array(contact)
        a = a[a[:, 1].argsort()]
        contact_points = [[a[0]]]

        for a0, a1 in zip(a[:-1], a[1:]):
            if a0[1] - a1[1] > 1:
                contact_points.append([a1])
            else:
                contact_points[-1].append(a1)

        contact_points = [np.array(c).mean(axis=0) for c in np.array(contact_points)]
        return contact_points
