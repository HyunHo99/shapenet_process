import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import pyrender
import trimesh

def generate_source(
        mesh_path,
        theta,
        phi,
        height,
        width,
        znear=0.05,
        zfar=1500,
):
    # theta : rotate horizontal, (0, pi) (y축과 이루는 각)
    # phi : rotate vertical (0, 2*pi) (x축과 이루는 각)
    # input = phi, theta of target camera, width, height
    # output = W*H*3 array
    fx = 39.227512 / 0.0369161
    fy = 39.227512 / 0.0369161
    K = np.array([fx, 0, width / 2,
                  0, fy, height / 2,
                  0, 0, 1]).reshape((3, 3))
    z = np.array([
        np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)
    ])
    print(z)
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
    scene = pyrender.Scene()
    tmesh = trimesh.load(mesh_path)
    scene = pyrender.Scene.from_trimesh_scene(tmesh)
    # mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    #
    # scene.add_node(mesh_node)

    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        znear=znear,
        zfar=100,
    )
    cam_node = pyrender.Node(camera=cam, matrix=T)
    scene.add_node(cam_node)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
    light_node = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light_node, parent_node=cam_node)
    pyrender.Viewer(scene)
    render = pyrender.OffscreenRenderer(width, height)
    color, depth = render.render(scene)

    plt.imshow(depth)
    plt.imshow(color)
    plt.show()
    # plt.savefig("depth.png")
    #
    # Pi = np.zeros((3, 4))
    # Pi[:, :3] = np.linalg.inv(K)
    # Pi[:, 3] = -z
    # Pi = R[:3, :3].T @ Pi
    # xyzs = []
    # for i in range(height):
    #     for j in range(width):
    #         dt = depth[i][j]
    #         if dt == 0:
    #             xyzs.append(np.array([0, 0, 0]))
    #             continue
    #         uvd = np.array([dt * (i + 0.5), dt * (j + 0.5), dt, 1])
    #         xyz = Pi @ uvd
    #         xyzs.append(xyz)
    # xyzs = np.array(xyzs).reshape(height, width, 3)
    # return xyzs


path = "./models2/models/model_normalized.obj"
generate_source(path, theta=np.pi/4, phi=np.pi/2, width=500, height=500)