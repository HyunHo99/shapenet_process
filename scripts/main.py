import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['EGL_DEVICE_ID'] = "1"
sys.path.append(".")
sys.path.append("..")

import argparse
import time
import multiprocessing as mp
from joblib import Parallel, delayed
from itertools import repeat

import cv2
import numpy as np

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

    # render and save the results
    for view_idx, (theta, phi) in enumerate(zip(thetas, phis)):
        img, depth, mask = render_mesh_o3d(
            mesh,
            theta,
            phi,
            height, width,
        )
    
        cv2.imwrite(os.path.join(sample_img_dir, "{}.jpg".format(view_idx)), img)
        cv2.imwrite(os.path.join(sample_depth_dir, "{}.exr".format(view_idx)), depth)
        cv2.imwrite(os.path.join(sample_mask_dir, "{}.jpg".format(view_idx)), mask)

def render_shapenet_samples(
    shapenet_src_dir: str,
    save_dir: str,
    height: int, width: int,
    ) -> None:

    # get sample directories
    sample_ids = [d for d in os.listdir(shapenet_src_dir)]

    # create the save directory
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    start_t = time.time()
    
    num_cores = mp.cpu_count() // 2
    print("[!] Using {} cores".format(num_cores))
    _ = Parallel(n_jobs=num_cores)(
        delayed(_render_shapenet_sample)(src_dir, sample_id, out_dir, H, W) \
            for src_dir, sample_id, out_dir, H, W in \
            zip(repeat(shapenet_src_dir), sample_ids, repeat(save_dir), repeat(height), repeat(width))
    )

    print("[!] Took {} seconds".format(time.time() - start_t))
    print("[!] Done.")

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapenet_path", type=str, default="./data")
    parser.add_argument("--save_path", type=str, default="./result")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    # render
    render_shapenet_samples(
        args.shapenet_path, "result", 
        args.height, args.width
        )
