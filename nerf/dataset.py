import numpy as np
import os
from PIL import Image

def load_cameras_txt(path):
    """
    Parse cameras.txt (COLMAP format) and return a dict:
      {camera_id: { 'model': str,
                    'width': int,
                    'height': int,
                    'params': np.array([...]) }}
    """
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            cam_id, model, w, h, *params = line.split()
            cams[int(cam_id)] = {
                'model': model,
                'width':  int(w),
                'height': int(h),
                'params': np.array(params, dtype=float)
            }
    return cams

def load_images_txt(path):
    """
    Parse images.txt (COLMAP format) and return:
      - poses: {image_id: {'qvec': np.array(4), 'tvec': np.array(3), 'camera_id': int}}
      - name_map: {image_id: filename}
    """
    poses = {}
    name_map = {}
    with open(path, 'r') as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    # images.txt alternates pose-line, then 2D-point-line; we only read every 2nd line
    for pose_line in lines[::2]:
        elems = pose_line.split()
        img_id = int(elems[0])
        qvec   = np.array(elems[1:5], dtype=float)
        tvec   = np.array(elems[5:8], dtype=float)
        cam_id = int(elems[8])
        fname  = elems[9]
        poses[img_id]    = {'qvec': qvec, 'tvec': tvec, 'camera_id': cam_id}
        name_map[img_id] = fname
    return poses, name_map

def load_colmap_dataset(colmap_dir, image_dir):
    """
    High-level loader: returns lists of dicts,
    each with image array + intrinsics + extrinsics.
    """
    cams = load_cameras_txt(os.path.join(colmap_dir, 'cameras.txt'))
    poses, name_map = load_images_txt(os.path.join(colmap_dir, 'images.txt'))
    data = []
    for img_id, info in poses.items():
        cam = cams[info['camera_id']]
        img_path = os.path.join(image_dir, name_map[img_id])
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        data.append({
            'image_id': img_id,
            'image': img,
            'intrinsics': cam,
            'qvec': info['qvec'],
            'tvec': info['tvec']
        })
    return data
