import sys
sys.path.append(".")
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import os
import pyrender
import trimesh
import random

from utils.math import *

def generate_source_depths(
        scene,
        height,
        width,
        znear=0.05,
        zfar=1500,
):
    # theta : rotate horizontal, (0, pi) (y축과 이루는 각)
    # phi : rotate vertical [0, 2*pi] (x축과 이루는 각)
    # input = phi, theta of target camera, width, height, pyrender scene of mesh
    # output = W*H*3 array
    theta = random.uniform(1e-5, np.pi-1e-5)
    phi = random.uniform(0, 2 * np.pi)

    # set camera intrinsics
    fx = 1062
    fy = 1062
    K = build_camera_intrinsic(fx, fy, height, width)
    
    # set camera extrinsics
    r = 3 * scene.scale
    E = build_camera_extrinsic(
        r, theta, phi,
        np.array([0., 1., 0.]),
    )

    # add camera
    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        znear=znear,
        zfar=zfar,
    )
    cam_node = pyrender.Node(camera=cam, matrix=E)
    scene.add_node(cam_node)

    # add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=cam_node)

    render = pyrender.OffscreenRenderer(width, height)
    _, depth = render.render(scene)
    render.delete()

    # normalize depth
    depth = depth / r
    plt.imshow(depth)
    plt.show()

    z = E[:3, 3]
    R = E[:3, :3]
    view_direction = - z / np.linalg.norm(z)

    Pi = np.zeros((3, 4))
    Pi[:, :3] = np.linalg.inv(K)
    Pi[:, 3] = -z
    Pi = R[:3, :3].T @ Pi

    return depth, view_direction, Pi
    