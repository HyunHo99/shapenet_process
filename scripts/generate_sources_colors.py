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

def generate_source_colors(
        scene,
        height,
        width,
        znear=0.05,
        zfar=1500,
):
    # theta : rotate horizontal, (0, pi] (y축과 이루는 각)
    # phi : rotate vertical [0, 2*pi] (x축과 이루는 각)
    # input = width, height, mesh_path(.obj file)
    # output = W*H*3 array

    # set camera intrinsic
    fx = 1062
    fy = 1062
    K = build_camera_intrinsic(fx, fy, height, width)

    # set camera extrinsic
    theta = random.uniform(1e-5, np.pi-1e-5)
    phi = random.uniform(0, 2 * np.pi)
    r = 3 * scene.scale
    E = build_camera_extrinsic(
        r, theta, phi,
        np.array([0., 1., 0.])
    )

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

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=cam_node)
    render = pyrender.OffscreenRenderer(width, height)
    color, _ = render.render(scene)
    render.delete()
    scene.remove_node(cam_node)

    z = E[:3, 3]
    view_direction = - z / np.linalg.norm(z)
    return color, view_direction
