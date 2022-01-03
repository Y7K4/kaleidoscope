"""The main script of the kaleidoscope.

The user can play around with the kaleidoscope. The original stage of the kaleidoscope is imported from a png file.
The kaleidoscope uses the mpm class to simulate the particle movements, and the reflection class to calculate the image.
An additional function is defined in the main script to convert the pariticles to the original pixels.

By hitting SPACE, you can switch between original/image pixels.
By hitting LEFT/RIGHT, you can change angular velocity.
By hitting ESCAPE, you can exit the program.
"""

import taichi as ti
from mpm import Mpm
from reflect import Reflect

ti.init(arch=ti.cpu)

# object parameters(constants)
res = (512, 512)  # resolution
center = ti.Vector([res[0] / 2.0, res[1] / 2.0])  # center of the pixels/mirrors
n_mirrors = 5  # number of mirrors for the kaleidoscope
mirror_radius = 0.15 * res[0]  # radius of the circumscribed circle
cap_shift = [0.0, 1.0/3.0]
cap_center = (ti.Vector([0.5, 0.5]) + ti.Vector(cap_shift))*res[0]  # center of the cap

# initialize objects
# XXX: Use the absolute path here if the relative path does not work
mpm = Mpm('obj.png')
kaleidoscope = Reflect(center, n_mirrors, mirror_radius, res)


@ti.kernel
def get_pixels():
    """calculate the pixels for kaleidoscpoe from the mpm update.
    """
    theta0 = mpm.theta[None]
    pi = 3.1415927
    #draw the background
    for i, j in kaleidoscope.obj_pixels:
        dpos = ti.Vector([i, j]) - cap_center
        r = dpos.norm()
        # white for inside the cap circle
        if r < res[0]/2:
            kaleidoscope.obj_pixels[i, j] = ti.Vector([1, 1, 1])
        # draw the boundary in dark gray, and light gray for outside the cap circle 
        else:
            theta = ti.acos(dpos.dot(ti.Vector([ti.cos(theta0), ti.sin(theta0)])) / r)
            if r - res[0]/2 < 10 and int((theta + pi / 60) / (pi / 30)) % 2 == 0:
                kaleidoscope.obj_pixels[i, j] = ti.Vector([0.2, 0.2, 0.2]) # draw the boundary
            else:
                kaleidoscope.obj_pixels[i, j] = ti.Vector([0.6, 0.6, 0.6]) # outside the cap circle

    # draw the particles
    for k in mpm.x:
        for i, j in ti.static(ti.ndrange(11, 11)):
            d_norm = ti.Vector([i - 5.0, j - 5.0]).norm(1e-3)
            if d_norm < 5.0:
                x = int(mpm.x[k][0] * res[0] + i - 5 + res[0] * cap_shift[0])
                y = int(mpm.x[k][1] * res[1] + j - 5 + res[1] * cap_shift[1])
                if x > 0 and x < res[0] and y > 0 and y < res[1]:
                    weight = 1 / (d_norm + 1.0) # weight higher when nearer to the particle position(center)
                    kaleidoscope.obj_pixels[x, y] = mpm.color[k] * weight + kaleidoscope.obj_pixels[x, y] * (1 - weight)

gui = ti.GUI('Kaleidoscope', res=res)
gui.fps_limit = 30
show_obj = False
omega = 4
while gui.running:
    # get keyboard event
    for e in gui.get_events(gui.PRESS, gui.MOTION):
        if e.key == ti.GUI.SPACE:
            show_obj = not show_obj
        elif e.key == ti.GUI.RIGHT:
            if omega < 4:
                omega += 1
        elif e.key == ti.GUI.LEFT:
            if omega > -4:
                omega -= 1
        elif e.key == ti.GUI.ESCAPE:
            exit()

    # mpm simulator
    mpm.step(2e-3, omega)
    # transfer particle data to pixels
    get_pixels()
    # get image from object
    kaleidoscope.update_img()
    # show image
    if show_obj:
        gui.set_image(kaleidoscope.obj_pixels)
        for k in range(n_mirrors):
            gui.line(kaleidoscope.mirror_pts[k].to_numpy() / res[0],
                     kaleidoscope.mirror_pts[k + 1].to_numpy() / res[0], color=0)
    else:
        gui.set_image(kaleidoscope.img_pixels)

    # print rules with user-friendly color
    gui.rect(topleft=[0.1, 0.17], bottomright=[0.9, 0.1], radius=25, color=16777215)
    gui.text("Current angular velocity: " + str(round(omega, 2)), pos=(0.1, 0.2), color=0)
    gui.text("Press SPACE to switch between original/image pixels", pos=(0.1, 0.15), color=0)
    gui.text("Press LEFT/RIGHT to change angular velocity.", pos=(0.1, 0.1), color=0)
    gui.show()
