import taichi as ti
from random import random

# reference: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm99.py


@ti.data_oriented
class Mpm:
    def __init__(self, filepath, quality=1):
        """[summary]

        Args:
            quality (int, optional): [description]. Defaults to 1.
        """
        # material types
        self.MATERIAL_LIQUID = 0
        self.MATERIAL_JELLY = 1
        self.MATERIAL_SNOW = 2

        # constants
        self.particle_count = 20000 * quality**2
        self.grid_res = 128 * quality
        self.dx = 1 / self.grid_res
        self.inv_dx = float(self.grid_res)
        self.dt = 1e-4 / quality
        self.p_vol = (self.dx * 0.5)**2
        self.p_rho = 1
        self.p_mass = self.p_vol * self.p_rho
        self.E = 1.5e4  # Young's modulus
        self.nu = 0.4  # Poisson's ratio
        self.mu_0 = self.E / (2 * (1 + self.nu))  # Lame parameters
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) / (1 - 2 * self.nu))  # Lame parameters
        self.g = ti.Vector([0.0, -9.8])  # gravitational acceleration
        self.mu_boundary = 0.8  # mu at boundary

        # particles
        self.x = ti.Vector.field(2, float, self.particle_count)  # position
        self.v = ti.Vector.field(2, float, self.particle_count)  # velocity
        self.C = ti.Matrix.field(2, 2, float, self.particle_count)  # affine velocity
        self.F = ti.Matrix.field(2, 2, float, self.particle_count)  # deformation gradient
        self.material = ti.field(int, self.particle_count)  # material type
        self.color = ti.Vector.field(3, float, self.particle_count)  # color
        self.color_id = ti.field(float, self.particle_count)  # color id
        self.palette = []  # list of colors (hex)

        # grids
        self.Jp = ti.field(float, self.particle_count)  # plastic deformation
        self.grid_m = ti.field(float, (self.grid_res, self.grid_res))  # grid node mass
        self.grid_v = ti.Vector.field(2, float, (self.grid_res, self.grid_res))  # grid node momentum

        # load initial state from image
        pixels = ti.imread(filepath)
        roi_size = min(pixels.shape[:1])
        for p in range(self.particle_count):
            while True:
                x = random()
                y = random()
                color_rgb = pixels[int(x * roi_size)][int(y * roi_size)]
                color_hex = color_rgb.dot([65536, 256, 1])
                if color_hex != 0xffffff:  # if not white
                    if color_hex not in self.palette:
                        self.palette.append(color_hex)
                    self.x[p] = [x, y]
                    self.F[p] = ti.Matrix([[1, 0], [0, 1]])
                    self.Jp[p] = 1
                    self.material[p] = 1
                    self.color_id[p] = self.palette.index(color_hex)
                    for channel in range(3):
                        self.color[p][channel] = color_rgb[channel] / 255.0
                    break

    def step(self, timestep, omega):
        for _ in range(int(timestep // self.dt)):
            self.substep(omega)

    @ti.kernel
    def substep(self, omega: float):
        # reset grids
        for i, j in self.grid_m:
            self.grid_m[i, j] = 0
            self.grid_v[i, j] = [0, 0]

        # particle to grid (P2G)
        for p in self.x:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]  # quadratic kernel
            self.F[p] = (ti.Matrix.identity(float, 2) + self.dt * self.C[p]) @ self.F[p]
            h = ti.exp(10 * (1.0 - self.Jp[p]))  # hardening coefficient: snow gets harder when compressed
            if self.material[p] == self.MATERIAL_JELLY:
                h = 0.4
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.MATERIAL_LIQUID:
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if self.material[p] == self.MATERIAL_SNOW:
                    new_sig = min(max(sig[d, d], 1 - 2.5e2), 1 + 4.5e-3)  # plasticity
                self.Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.material[p] == self.MATERIAL_LIQUID:
                # reset deformation gradient to avoid numerical instability
                self.F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif self.material[p] == self.MATERIAL_SNOW:
                # reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (self.F[p] - U @ V.transpose()) @ self.F[p].transpose() + \
                ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

        # grid operations
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0:
                self.grid_v[i, j] /= self.grid_m[i, j]
                self.grid_v[i, j] += self.dt * self.g  # gravity
                # limit inside box
                if i < 3 and self.grid_v[i, j][0] < 0:
                    self.grid_v[i, j][0] = 0
                if i > self.grid_res - 3 and self.grid_v[i, j][0] > 0:
                    self.grid_v[i, j][0] = 0
                if j < 3 and self.grid_v[i, j][1] < 0:
                    self.grid_v[i, j][1] = 0
                if j > self.grid_res - 3 and self.grid_v[i, j][1] > 0:
                    self.grid_v[i, j][1] = 0
                # limit inside circle
                dpos = ti.Vector([i * self.dx - 0.5, j * self.dx - 0.5])
                if dpos.norm() >= 0.5 and dpos.dot(self.grid_v[i, j]) > 0:
                    v_radial = dpos.dot(self.grid_v[i, j]) / dpos.norm()
                    v_tangential = dpos.cross(self.grid_v[i, j]) / dpos.norm()
                    if omega > v_tangential:
                        v_tangential = ti.min(omega, v_tangential + self.mu_boundary * v_radial)
                    else:
                        v_tangential = ti.max(omega, v_tangential - self.mu_boundary * v_radial)
                    self.grid_v[i, j] = ti.Matrix([[0, -1], [1, 0]]) @ dpos / dpos.norm() * v_tangential

        # grid to particle (G2P)
        for p in self.x:
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt * self.v[p]  # advection


if __name__ == '__main__':
    ti.init(arch=ti.cpu)

    mpm = Mpm('obj.png')

    gui = ti.GUI("Kaleidoscope MPM", res=512, background_color=0x000000)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        mpm.step(2e-3, -2)
        gui.circle([0.5, 0.5], 0xffffff, 256)
        gui.circles(mpm.x.to_numpy(),
                    radius=5,
                    palette=mpm.palette,
                    palette_indices=mpm.color_id)
        gui.show()
