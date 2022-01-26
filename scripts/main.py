import sys
sys.path.append("..")

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyrender

from utils.math import *

def render_mesh_o3d(
        mesh_path: str,
        save_path: str,
        theta: float,
        phi: float,
        height: int,
        width: int,
        znear: float = 0.01,
        zfar: float = 1500,
    ) -> None:
    """
    Renders a mesh loaded as open3d.geometry.TriangleMesh object.

    Args:
    - mesh_path: String.
        Path to where the mesh is stored.
    - save_path: String.
        Path to where the rendered image will be saved.
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
    """
    # set camera intrinsics
    fx = 39.227512 / 0.0369161
    fy = 39.227512 / 0.0369161
    K = build_camera_intrinsic(fx, fy, height, width)
    
    # set camera extrinsics
    E = build_camera_extrinsic(
        1.0, theta, phi,
        np.array([0., 1., 0.])
    )

    # load mesh and process
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.7, 0.7, 0.7))
    box = mesh.get_axis_aligned_bounding_box()
    mesh_scale = ((box.get_max_bound() - box.get_min_bound()) ** 2).sum()

    mesh = mesh.scale(0.3 * mesh_scale, center=(0, 0, 0))
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

    render = pyrender.OffscreenRenderer(width, height)
    color, depth = render.render(scene)

    plt.imshow(depth)
    plt.show()

    plt.imshow(color)
    plt.show()


if __name__ == "__main__":
    path = "./models2/models/model_normalized.obj"
    out_file = "./depth.png"
    render_mesh_o3d(
        path, 
        out_file, 
        theta=np.pi/3, 
        phi=-np.pi/2, 
        width=500, height=500
    )

