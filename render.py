# render.py - Functions used for rendering ShapeNet models

import sys

sys.path.append(".")
sys.path.append("..")

import open3d as o3d
import pyrender
import matplotlib.pyplot as plt

from utils.math import *


def render_mesh_o3d(
        mesh: o3d.geometry.TriangleMesh,
        theta: float,
        phi: float,
        height: int,
        width: int,
        znear: float = 0.01,
        zfar: float = 1500,
    ) -> Tuple[np.array, np.array, np.array]:
    """
    Renders a mesh loaded as open3d.geometry.TriangleMesh object.

    Args:
    - mesh_path: Open3D.geometry.TriangleMesh object.
        A mesh to be rendered.
    - theta: Float.
        Angle between positive direction of y axis and displacement vector.
    - phi: Float.
        Angle between positive direction of x axis and displacement vector.
    - height: Int.
        Height of the viewport.
    - width: Int.
        Width of the viewport.
    - znear: Float.
        The nearest visible depth.
    - zfar: Float.
        The farthest visible depth.

    Returns:
    - color: A Numpy array of shape (3, height, width).
    - depth: A Numpy array of shape (1, height, width).
    - mask: A Numpy array of shape (1, height, width).
    - K: A Numpy array of shape (3, 4).
    - E: A Numpy array of shape (3, 4).
    """
    # set camera intrinsics
    fx = 39.227512 / 0.0369161
    fy = 39.227512 / 0.0369161
    K = build_camera_intrinsic(fx, fy, height, width)
    
    # set camera extrinsics
    E = build_camera_extrinsic(
        1.2, theta, phi,
        np.array([0., 1., 0.])
    )

    # parse mesh data
    verts = np.asarray(mesh.vertices).astype(np.float32)
    faces = np.asarray(mesh.triangles).astype(np.int32)
    colors = np.asarray(mesh.vertex_colors).astype(np.float32)
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)

    scene = pyrender.Scene()

    # add mesh
    mesh = pyrender.Mesh(
        primitives=[
            pyrender.Primitive(
                positions=verts,
                normals=normals,
                color_0=colors,
                indices=faces,
                mode=pyrender.GLTF.TRIANGLES,
            )
        ],
        is_visible=True,
    )
    mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
    scene.add_node(mesh_node)

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

    # render
    render = pyrender.OffscreenRenderer(width, height)
    img, depth = render.render(scene)
    depth[depth == 0.0] = np.inf
    mask = ~np.isinf(depth)
    mask = (mask.astype(np.uint8) * 255).astype(np.uint8)

    return img, depth, mask, K[:3, :], E[:3, :]
