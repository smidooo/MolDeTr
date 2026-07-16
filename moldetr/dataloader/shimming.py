'''
Minimal SHIMpanzee code to simulate collate

modifed from https://github.com/smeerten/shimpanzee under GNU GPL licence
'''
import numpy as np
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

# Maximum shim value limits
Z1LIM, X1LIM, Y1LIM = 20.0, 20.0, 20.0
Z2LIM, XZLIM, XYLIM, YZLIM, X2_Y2LIM = 5.0, 10.0, 10.0, 10.0, 10.0
Z3LIM, XZ2LIM, YZ2LIM, ZX2_ZY2LIM, XYZLIM, X3LIM, Y3LIM = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

NSTEPS = 1000


class ShimSim:
    def __init__(self, N=100000, dimensions=[2.5, 10], angles=[0, 0], sw=500.0, npoints=512, sphere=False):
        self.OVERSAMPLE = 16
        self.N, self.r, self.l, self.theta, self.phi, self.sw, self.npoints, self.sphere = N, dimensions[0], dimensions[
            1], angles[0], angles[1], sw, npoints, sphere
        self.fidSurface = 0.0
        self.fwhm = 0.0
        self.reset()

    def setupGrid(self):
        # create 3D "volume"
        if self.sphere:
            phi = np.random.uniform(0, 2 * np.pi, self.N)
            costheta = np.random.uniform(-1, 1, self.N)
            u = np.random.uniform(0, 1, self.N)
            theta = np.arccos(costheta)
            R = self.r * np.cbrt(u)
            self.x = R * np.sin(theta) * np.cos(phi)
            self.y = R * np.sin(theta) * np.sin(phi)
            self.z = R * np.cos(theta)
        else:
            self.z = np.random.uniform(-self.l / 2.0, self.l / 2.0, self.N)
            R = np.random.triangular(0, self.r, self.r, self.N)
            angle = np.random.uniform(0, 2 * np.pi, self.N)
            self.x = R * np.cos(angle)
            self.y = R * np.sin(angle)
        # ShimTypes
        # Values given by spherical harmonics' coefficients
        self.Y = np.sqrt(3 / (4 * np.pi)) * self.y
        self.X = np.sqrt(3 / (4 * np.pi)) * self.x
        self.Z = np.sqrt(3 / (4 * np.pi)) * self.z
        self.XY = 0.5 * np.sqrt(15 / np.pi) * self.x * self.y
        self.YZ = 0.5 * np.sqrt(15 / np.pi) * self.y * self.z
        self.Z2 = 0.25 * np.sqrt(5 / np.pi) * (2 * self.z ** 2 - self.x ** 2 - self.y ** 2)
        self.XZ = 0.5 * np.sqrt(15 / np.pi) * self.x * self.z
        self.X2_Y2 = 0.25 * np.sqrt(15 / np.pi) * (self.x ** 2 - self.y ** 2)
        self.Y3 = 0.25 * np.sqrt(35 / (2 * np.pi)) * (3 * self.x ** 2 * self.y - self.y ** 3)
        self.XYZ = 0.5 * np.sqrt(105 / np.pi) * self.z * self.x * self.y
        self.YZ2 = 0.25 * np.sqrt(21 / (2 * np.pi)) * self.y * (4 * self.z ** 2 - self.x ** 2 - self.y ** 2)
        self.Z3 = 0.25 * np.sqrt(7 / np.pi) * self.z * (2 * self.z ** 2 - 3 * self.x ** 2 - 3 * self.y ** 2)
        self.XZ2 = 0.25 * np.sqrt(21 / (2 * np.pi)) * self.x * (4 * self.z ** 2 - self.x ** 2 - self.y ** 2)
        self.ZX2_ZY2 = 0.25 * np.sqrt(105 / np.pi) * self.z * (self.x ** 2 - self.y ** 2)
        self.X3 = 0.25 * np.sqrt(35 / (2 * np.pi)) * (self.x ** 3 - 3 * self.y ** 2 * self.x)

        self.Mfield = np.zeros(self.x.shape)
        self.freq = np.linspace(-self.sw / 2.0, self.sw / 2.0, self.npoints)
        self.lb = np.exp(
            -((10 * np.pi * np.arange(self.npoints * self.OVERSAMPLE) / self.sw) ** 2) / (4.0 * np.log(2))) / self.N
        self.lb[self.npoints - (self.npoints + 1) // 2:] = 0
        self.lb[0] = self.lb[0] / 2.0
        self.lbScale = np.sum(np.abs(self.lb)) * self.N / (self.npoints * self.OVERSAMPLE * 100.0)

    def simulate(self, z1=0, x1=0, y1=0, z2=0, xz=0, yz=0, x2_y2=0, xy=0, z3=0, xz2=0, yz2=0, zx2_zy2=0, xyz=0, x3=0,
                 y3=0):
        y1 += self.y1Game
        z1 += self.z1Game
        x1 += self.x1Game
        xy += self.xyGame
        yz += self.yzGame
        z2 += self.z2Game
        xz += self.xzGame
        x2_y2 += self.x2_y2Game
        y3 += self.y3Game
        xyz += self.xyzGame
        yz2 += self.yz2Game
        z3 += self.z3Game
        xz2 += self.xz2Game
        zx2_zy2 += self.zx2_zy2Game
        x3 += self.x3Game
        self.Mfield = z1 * self.Z + z2 * self.Z2 + z3 * self.Z3
        self.Mfield += x1 * self.X + y1 * self.Y
        self.Mfield += xz * self.XZ + xy * self.XY + yz * self.YZ + x2_y2 * self.X2_Y2
        self.Mfield += xz2 * self.XZ2 + yz2 * self.YZ2 + zx2_zy2 * self.ZX2_ZY2 + xyz * self.XYZ + x3 * self.X3 + y3 * self.Y3
        # Spectrum is just a histogram over all "frequencies" in inhomogeneous volume
        self.spectrum, _ = np.histogram(self.Mfield, self.npoints * self.OVERSAMPLE,
                                        (-self.sw / 2.0, self.sw / 2.0))  # (array,bins,range)
        self.spectrum = np.fft.ifft(self.spectrum) * self.lb
        self.fidSurface = (np.sum(np.abs(self.spectrum))) / (self.lbScale)
        self.spectrum = np.fft.fft(self.spectrum)[::self.OVERSAMPLE]
        self.spectrum = np.real(self.spectrum)
        self.spectrum = self.spectrum * 1000  # modified! Scale by 1000 to avoid vanishing gradients
        r = UnivariateSpline(self.freq, self.spectrum - np.max(self.spectrum) / 2.0, s=0).roots()
        self.fwhm = np.abs(max(r) - min(r))
        self.peak_max = np.max(self.spectrum)

    def resetGame(self):
        self.y1Game = 0.0
        self.z1Game = 0.0
        self.x1Game = 0.0
        self.xyGame = 0.0
        self.yzGame = 0.0
        self.z2Game = 0.0
        self.xzGame = 0.0
        self.x2_y2Game = 0.0
        self.y3Game = 0.0
        self.xyzGame = 0.0
        self.yz2Game = 0.0
        self.z3Game = 0.0
        self.xz2Game = 0.0
        self.zx2_zy2Game = 0.0
        self.x3Game = 0.0

    def startGame(self, order=4, zonly=False):
        self.resetGame()
        if zonly:
            self.z1Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Z1LIM / NSTEPS
            self.z2Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Z2LIM / NSTEPS
            self.z3Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Z3LIM / NSTEPS
            return
        self.z1Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Z1LIM / NSTEPS
        self.x1Game = np.random.randint(-NSTEPS, NSTEPS + 1) * X1LIM / NSTEPS
        self.y1Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Y1LIM / NSTEPS
        if order > 1:
            self.z2Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Z2LIM / NSTEPS
            self.xzGame = np.random.randint(-NSTEPS, NSTEPS + 1) * XZLIM / NSTEPS
            self.yzGame = np.random.randint(-NSTEPS, NSTEPS + 1) * YZLIM / NSTEPS
            self.x2_y2Game = np.random.randint(-NSTEPS, NSTEPS + 1) * X2_Y2LIM / NSTEPS
            self.xyGame = np.random.randint(-NSTEPS, NSTEPS + 1) * XYLIM / NSTEPS
        if order > 2:
            self.z3Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Z3LIM / NSTEPS
            self.xz2Game = np.random.randint(-NSTEPS, NSTEPS + 1) * XZ2LIM / NSTEPS
            self.yz2Game = np.random.randint(-NSTEPS, NSTEPS + 1) * YZ2LIM / NSTEPS
            self.zx2_zy2Game = np.random.randint(-NSTEPS, NSTEPS + 1) * ZX2_ZY2LIM / NSTEPS
            self.xyzGame = np.random.randint(-NSTEPS, NSTEPS + 1) * XYZLIM / NSTEPS
            self.x3Game = np.random.randint(-NSTEPS, NSTEPS + 1) * X3LIM / NSTEPS
            self.y3Game = np.random.randint(-NSTEPS, NSTEPS + 1) * Y3LIM / NSTEPS

    def reset(self):
        self.resetGame()
        self.setupGrid()


def simulate_data(nr_samples=5, npoints=20):
    sim = ShimSim(npoints=npoints)
    sim.startGame()
    data_all = []
    labels_all = np.empty([nr_samples, 4])
    for i in range(nr_samples):
        sim.reset()
        sim.startGame()
        samples_list = []
        for k in range(4):
            trial = np.random.uniform(-1, 1, 4)
            sim.simulate(z1=trial[0] * Z1LIM, z2=trial[1] * Z2LIM, x1=trial[2] * X1LIM, y1=trial[3] * Y1LIM)
            sample = {'spectrum': sim.spectrum, 'trial': trial, 'fwhm': sim.fwhm * (npoints / 512),
                      'peak_max': sim.peak_max, 'fidSurface': sim.fidSurface}
            samples_list.append(sample)
        data_all.append(samples_list)
        labels_all[i] = np.array([sim.z1Game, sim.z2Game, sim.x1Game, sim.y1Game]) / [Z1LIM, Z2LIM, X1LIM, Y1LIM]

    return data_all, labels_all


if __name__ == "__main__":

    # Assume simulate_data is defined as per the previous modification
    data, labels = simulate_data()

    # Iterate over each sample's collate and labels
    for sample_data, sample_label in zip(data, labels):
        # Each item in sample_data is a dictionary with 'spectrum' and 'trial'
        for trial_data in sample_data:
            spectrum = trial_data['spectrum']
            plt.plot(spectrum, label=f"spectrum with fwhm: {trial_data['fwhm']:.2f}")
            plt.title(
                f"z1: {sample_label[0]:.2f}, z2: {sample_label[1]:.2f}, x1: {sample_label[2]:.2f}, y1: {sample_label[3]:.2f}")
            plt.legend()
            plt.show()
