import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = "1"
sys.path.append(".")
sys.path.append("..")

import argparse
import csv
import time

import multiprocessing as mp
from joblib import Parallel, delayed
from itertools import repeat

import cv2
import numpy as np
from tqdm import tqdm

from render import *
from utils.math import *

def _render_shapenet_sample(
    shapenet_src_dir: str, sample_id: str, 
    save_dir: str,
    height: int, width: int,
    ) -> None:
    sample_dir = os.path.join(shapenet_src_dir, str(sample_id))
    assert os.path.exists(sample_dir), "[!] Path {} does not exist".format(sample_dir)

    # if not a directory, ignore and quit
    if not os.path.isdir(sample_dir):
        return

    # specify the file name of the mesh
    # if mesh file is missing, ignore and quit
    mesh_file = os.path.join(sample_dir, "models/model_normalized.obj")
    if not os.path.exists(mesh_file):
        return

    # create directories for outputs
    sample_save_dir = os.path.join(save_dir, str(sample_id))
    sample_img_dir = os.path.join(sample_save_dir, "image")
    sample_depth_dir = os.path.join(sample_save_dir, "depth")
    sample_mask_dir = os.path.join(sample_save_dir, "mask")

    if not os.path.exists(sample_save_dir):
        os.mkdir(sample_save_dir)
    if not os.path.exists(sample_img_dir):
        os.mkdir(sample_img_dir)
    if not os.path.exists(sample_depth_dir):
        os.mkdir(sample_depth_dir)
    if not os.path.exists(sample_mask_dir):
        os.mkdir(sample_mask_dir)

    # specify camera intrinsics and extrinsics
    phis = np.linspace(0, 2 * np.pi, 24)  # divide 360 degrees into 24 steps
    thetas = (np.pi / 2.05) * np.ones_like(phis)  # fixed elevation

    # load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.7, 0.7, 0.7))
    box = mesh.get_axis_aligned_bounding_box()
    mesh_scale = ((box.get_max_bound() - box.get_min_bound()) ** 2).sum()
    mesh = mesh.scale(0.35 * mesh_scale, center=(0, 0, 0))

    #mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #    size=0.1, origin=[0, 0, 0]
    #)
    #o3d.visualization.draw_geometries([mesh, mesh_frame])

    camera_params = {}

    # render and save the results
    for view_idx, (theta, phi) in enumerate(zip(thetas, phis)):
        img, depth, mask, K, E = render_mesh_o3d(
            mesh,
            theta,
            phi,
            height, width,
        )
    
        cv2.imwrite(os.path.join(sample_img_dir, "{}.jpg".format(view_idx)), img)
        cv2.imwrite(os.path.join(sample_depth_dir, "{}.exr".format(view_idx)), depth)
        cv2.imwrite(os.path.join(sample_mask_dir, "{}.jpg".format(view_idx)), mask)

        camera_params["world_mat_{}".format(view_idx)] = E
        camera_params["camera_mat_{}".format(view_idx)] = K
    np.savez(os.path.join(sample_mask_dir, "cameras.npz"), **camera_params)

def render_shapenet_samples(
    shapenet_src_dir: str,
    save_dir: str,
    height: int, width: int,
    sample_csv: str = None,
    ) -> None:

    # get sample directories
    if sample_csv is None:
        sample_ids = [d for d in os.listdir(shapenet_src_dir)]
    else:
        with open(sample_csv, "r", encoding="utf-8") as f:
            content = csv.reader(f)

            sample_ids = []

            for idx, line in enumerate(content):
                if idx != 0:
                    fullID = line[0]
                    sample_id = fullID.split(".")[-1]
                    sample_ids.append(sample_id)

    # create the save directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    start_t = time.time()

    for sample_id in tqdm(sample_ids):
        _render_shapenet_sample(
            shapenet_src_dir,
            sample_id,
            save_dir,
            height, width
        )

    print("[!] Took {} seconds".format(time.time() - start_t))
    print("[!] Done.")

if __name__ == "__main__":
    if sys.platform == "darwin":
        print(
            "[!] Pyrender yields slightly different projection matrix on macOS. \
            We highly recommend you to run this script on other OS such as Linux, Windows, etc. \
            For details of problematic behavior of Pyrender, please refer to \
            https://pyrender.readthedocs.io/en/latest/_modules/pyrender/camera.html#IntrinsicsCamera.get_projection_matrix."
        )
        quit(-1)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_path", type=str, default="/home/dreamy1534/encoder4editing/data/paintme/02958343")
    parser.add_argument("--sample_csv", type=str, default="/home/dreamy1534/ShapeNet_Pyrender/sedan.csv", help="CSV holding IDs samples to be rendered")
    parser.add_argument("--save_path", type=str, default="/home/dreamy1534/encoder4editing/data/paintme/shapenet_sedan/")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    # render
    render_shapenet_samples(
        args.shapenet_path, 
        args.save_path, 
        args.height, args.width,
        args.sample_csv,
    )
