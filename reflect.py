"""Provides a Reflection class.

Define a group of mirrors that forms a regular polygon as the basic components of a kaledoscope. 
The eye position is assumed to be at the center of the polygon. 
The class provides a tracing function to trace the original object point from an image point,
and using the tracing function, it calculates the image of the object observed by the eye in the kaledoscope.
"""

#from os import cpu_count
import taichi as ti
import numpy as np

@ti.data_oriented
class Reflect:
    def __init__(self, center, n_mirrors, radius, res) -> None:
        """Initializes an Kaleidoscope Reflection object.
        Args:
            center (pair of int): Center of the mirrors.
            n_mirrors (int): number of mirrors.
            radius (float): radius of the circumscribed circle of the mirrors.
            res (pair of int): resolution
        """
        # 2d rigid body object-related field
        self.center = center # center of kaleidoscope
        self.n_mirrors = n_mirrors # number of edges of mirrors
        self.mirror_pts = ti.Vector.field(2, float, self.n_mirrors + 1)
        theta = np.linspace(0, 2 * np.pi, self.n_mirrors + 1)
        self.mirror_pts.from_numpy(np.stack((
            self.center[0] + radius * np.cos(theta),
            self.center[1] + radius * np.sin(theta)
            ), axis=-1))
        self.obj_pixels = ti.Vector.field(3, float, res)
        self.img_pixels = ti.Vector.field(3, float, res)

    @staticmethod
    @ti.func
    def reflection(obj, mirror_p0, mirror_p1):
        """Calculate the image position of an object.
        Args:
            obj (ti.Vector): the position of the object.
            mirror_p0 (ti.Vector): one point on the mirror.
            mirror_p1 (ti.Vector): another point on the mirror.
        """
        dx, dy = mirror_p1 - mirror_p0
        a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
        b = (2 * dx * dy) / (dx * dx + dy * dy)
        img = mirror_p0 + ti.Matrix([[a, b], [b, -a]]) @ (obj - mirror_p0)
        return img

    @staticmethod
    @ti.func
    def intersection(a0, a1, b0, b1):
        """Calculate the intersection points of two segments. Return a1 as default.
        Args:
            a0 (ti.Vector): one edge of one segment.
            a1 (ti.Vector): the other edge of the segment.
            b0 (ti.Vector): one edge of the other segment.
            b1 (ti.Vector): the other edge of the other segment.
        """
        ans = a1 # default to a1 if no intersection
        det = (a0 - a1).cross(b0 - b1)
        if det != 0:
            t_a = (a0 - b0).cross(b0 - b1) / det
            t_b = (a0 - b0).cross(a0 - a1) / det
            if 0 <= t_a <= 1 and 0 <= t_b <= 1: # within both segments
                ans = b0 + t_b * (b1 - b0)
        return ans

    @ti.func
    def tracing(self, i, j):
        """trace the original object position of an image position.
        Args:
            i (int): the x axis pixel.
            j (int): the y axis pixel.
        """
        src = self.center
        dst = ti.Vector([float(i), float(j)])
        for t in range(100): # in case it does not stop, set a maximum step
            closest_hit = dst
            closest_k = 0
            for k in ti.static(range(self.n_mirrors)):
                hit = self.intersection(src, dst, self.mirror_pts[k], self.mirror_pts[k + 1])
                if (hit - src).dot(hit - closest_hit) < -1e-1:
                    closest_hit = hit
                    closest_k = k
            if (closest_hit - dst).norm() < 1e-3:
                break
            src = closest_hit
            dst = self.reflection(dst, self.mirror_pts[closest_k], self.mirror_pts[closest_k + 1])
        return dst

    @ti.kernel
    def update_img(self):
        """update the image pixels from the object pixels.
        """
        for i, j in self.img_pixels:
            p = self.tracing(i, j)
            i_obj = p[0]
            j_obj = p[1]
            self.img_pixels[i, j] = self.obj_pixels[int(i_obj), int(j_obj)]
