from trajectory import MatlabFolder
import scipy.io as sio
import numpy as np
from copy import deepcopy
from os import path


class Ants_Frame:
    def __init__(self, position, angle, carrying, major_axis_length):
        self.position = position
        self.angle = angle
        self.carrying = carrying
        self.major_axis_length = major_axis_length

    def carrierCount(self):
        return len([pos for [pos, car] in zip(self.position, self.carrying) if car])


class Ants:
    def __init__(self, x):
        self.filename = x.filename
        self.VideoChain = [x.old_filenames(0)]
        self.frames = []
        self.pix2cm = np.NaN

        self.matlab_ant_loading(x)

    def __add__(self, other):
        if self.filename != other.filename:
            print('ants don''t belong together!!')
        ants_combined = deepcopy(self)
        ants_combined.frames = ants_combined.frames + other.frames
        ants_combined.VideoChain = ants_combined.VideoChain + other.VideoChain
        return ants_combined

    def averageCarrierNumber(self):
        if len(self.frames):
            return np.ceil(np.mean([frame.carrierCount() for frame in self.frames]))
        else:
            return None

    def matlab_ant_loading(self, x):
        if not (self.VideoChain[0] == 'XLSPT_4280007_XLSpecialT_1_ants (part 3).mat'):
            if not path.isfile(MatlabFolder('ant', x.size, x.shape, x.free) + path.sep + self.VideoChain[0]):
                breakpoint()
            file = sio.loadmat(MatlabFolder('ant', x.size, x.shape, x.free) + path.sep + self.VideoChain[0])

            if 'Direction' not in file.keys() and x.shape.endswith('ASH'):
                # file['Direction'] = input('L2R or R2L  ')
                file['Direction'] = None

            if x.shape.endswith('ASH'):
                if 'R2L' == file['Direction']:
                    if x.shape == 'LASH':
                        x.shape = 'RASH'
                        x.filename.replace('LASH', 'RASH')
                        x.VideoChain = [name.replace('LASH', 'RASH') for name in self.VideoChain]

                    else:
                        x.shape = 'LASH'
                        x.filename.replace('RASH', 'LASH')
                        x.VideoChain = [name.replace('RASH', 'LASH') for name in self.VideoChain]

                    if x.angle_error[0] == 0:
                        if x.shape == 'LASH':
                            x.angle_error = [2 * np.pi * 0.11 + self.angle_error[0]]
                        if x.shape == 'RASH':
                            x.angle_error = [-2 * np.pi * 0.11 + self.angle_error[
                                0]]  # # For all the Large Asymmetric Hs I had 0.1!!! (I think, this is why I needed the
                            # error in the end... )

                        if x.shape == 'LASH' and self.size == 'XL':  # # It seems like the exit walls are a bit
                            # crooked, which messes up the contact tracking
                            x.angle_error = [2 * np.pi * 0.115 + self.angle_error[0]]
                        if x.shape == 'RASH' and self.size == 'XL':
                            x.angle_error = [-2 * np.pi * 0.115 + self.angle_error[0]]

            self.pix2cm = file['pix2cm']
            matlab_cell = file['ants']
            for Frame in matlab_cell:
                data = Frame[0]
                if data.size != 0:
                    position = data[:, 2:4]
                    angle = data[:, 5] * np.pi / 180 + x.angle_error[0]
                    carrying = data[:, 4]
                    major_axis_length = data[:, 6]
                    ants_frame = Ants_Frame(position, angle, carrying, major_axis_length)
                else:
                    ants_frame = Ants_Frame(np.array([]), np.array([]), [], [])

                self.frames.append(ants_frame)
        else:
            import h5py
            with h5py.File(
                    MatlabFolder(self.solver, self.size, self.shape, self.free) + path.sep + self.old_filename,
                    'r') as f:
                load_center = np.matrix.transpose(f['load_center'][:, :])

        return self
