import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import pyrender
import trimesh
import random

def generate_source_colors(
        mesh_path,
        height,
        width,
        images_per_mesh=10,
        znear=0.05,
        zfar=1500,
):
    # theta : rotate horizontal, (0, pi] (y축과 이루는 각)
    # phi : rotate vertical [0, 2*pi] (x축과 이루는 각)
    # input = phi, theta of target camera, width, height, mesh_path,
    # output = W*H*3 array
    theta = random.uniform(1e-5, np.pi-1e-5)
    phi = random.uniform(0, 2 * np.pi)

    tmesh = trimesh.load(mesh_path)
    scene = pyrender.Scene.from_trimesh_scene(tmesh)

    r = 3 * scene.scale
    fx = 39.227512 / 0.0369161  # focal length
    fy = 39.227512 / 0.0369161
    K = np.array([fx, 0, width / 2,
                  0, fy, height / 2,
                  0, 0, 1]).reshape((3, 3))
    z = r * np.array([
        np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)
    ])
    z_norm = (z / np.linalg.norm(z))
    up_vector = np.array([0, 1, 0])
    left_vector = np.cross(up_vector, z_norm)
    left_vector = left_vector / np.linalg.norm(left_vector)
    up_vector = np.cross(z_norm, left_vector)
    R = np.eye(4)
    R[:3, :3] = np.stack((left_vector, up_vector, z_norm), axis=-1)
    T = np.eye(4)
    T[:3, 3] = z
    T = T @ R

    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        znear=znear,
        zfar=zfar,
    )
    cam_node = pyrender.Node(camera=cam, matrix=T)
    scene.add_node(cam_node)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=cam_node)
    render = pyrender.OffscreenRenderer(width, height)
    color, depth = render.render(scene)
    # plt.imshow(depth)
    # plt.show()
    plt.imshow(color)
    plt.show()


path = "./models2/models/model_normalized.obj"
generate_source_colors(path, images_per_mesh=10, width=500, height=500)