import os
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


PRIMITIVE_CLASSES = {
    "box": 1,
    "sphere": 2,
    "cylinder": 3,
    "torus": 4,
    "cone": 5,
}


def create_primitive(primitive_type,
                     size_range=(0.2, 0.6),
                     height_range=(0.2, 0.8)):
    s_min, s_max = size_range
    h_min, h_max = height_range

    if primitive_type == "box":
        sx = np.random.uniform(s_min, s_max)
        sy = np.random.uniform(s_min, s_max)
        sz = np.random.uniform(s_min, s_max)
        mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)

    elif primitive_type == "sphere":
        r = np.random.uniform(s_min, s_max) * 0.5
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)

    elif primitive_type == "cylinder":
        r = np.random.uniform(s_min, s_max) * 0.3
        h = np.random.uniform(h_min, h_max)
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h)

    elif primitive_type == "torus":
        r_torus = np.random.uniform(s_min, s_max) * 0.3
        r_tube = r_torus * 0.3
        mesh = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=r_torus,
            tube_radius=r_tube
        )

    elif primitive_type == "cone":
        r = np.random.uniform(s_min, s_max) * 0.4
        h = np.random.uniform(h_min, h_max)
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=r, height=h)

    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    mesh.compute_vertex_normals()
    color = np.random.uniform(0.2, 0.9, size=3)
    mesh.paint_uniform_color(color)

    return mesh


def random_transform(mesh,
                     scene_extent=2.0,
                     max_rotation_deg=45.0):
    tx = np.random.uniform(-scene_extent, scene_extent)
    ty = np.random.uniform(-scene_extent, scene_extent)
    tz = np.random.uniform(-scene_extent * 0.1, scene_extent * 0.1)

    angle = np.deg2rad(np.random.uniform(-max_rotation_deg, max_rotation_deg))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    R = np.array([
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz])

    mesh_t = mesh.transform(T)
    return mesh_t


def sample_points_from_mesh(mesh,
                            num_points=4096):
    pcd = mesh.sample_points_poisson_disk(num_points)
    if not pcd.has_colors():
        colors = np.ones((np.asarray(pcd.points).shape[0], 3), dtype=np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_scene(num_objects,
                points_per_object,
                scene_extent,
                primitive_types):
    all_points = []
    all_colors = []
    all_semantic = []
    all_instance = []
    bboxes = []

    for inst_id in range(num_objects):
        primitive_type = np.random.choice(primitive_types)
        mesh = create_primitive(primitive_type)
        mesh = random_transform(mesh, scene_extent=scene_extent)

        pcd = sample_points_from_mesh(mesh, num_points=points_per_object)

        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float32)

        all_points.append(pts)
        all_colors.append(cols)

        class_id = PRIMITIVE_CLASSES[primitive_type]
        semantic_labels = np.full(pts.shape[0], class_id, dtype=np.int32)
        instance_labels = np.full(pts.shape[0], inst_id, dtype=np.int32)

        all_semantic.append(semantic_labels)
        all_instance.append(instance_labels)

        aabb = pcd.get_axis_aligned_bounding_box()
        center = aabb.get_center()
        extent = aabb.get_extent()
        bbox = np.zeros(7, dtype=np.float32)
        bbox[0:3] = center
        bbox[3:6] = extent
        bbox[6] = class_id
        bboxes.append(bbox)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_semantic = np.concatenate(all_semantic, axis=0)
    all_instance = np.concatenate(all_instance, axis=0)
    bboxes = np.stack(bboxes, axis=0) if len(bboxes) > 0 else np.zeros((0, 7), dtype=np.float32)

    vert = np.concatenate([all_points, all_colors], axis=1)

    return vert, all_semantic, all_instance, bboxes


def generate_dataset(output_dir,
                     num_train=50,
                     num_val=10,
                     num_test=10,
                     objects_per_scene=(3, 6),
                     points_per_object=2048,
                     scene_extent=2.0,
                     primitive_types=None,
                     seed=42):
    if primitive_types is None:
        primitive_types = list(PRIMITIVE_CLASSES.keys())

    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }

    split_files = {
        "train": open(output_dir / "train_scenes.txt", "w"),
        "val": open(output_dir / "val_scenes.txt", "w"),
        "test": open(output_dir / "test_scenes.txt", "w"),
    }

    global_scene_idx = 0

    for split_name, num_scenes in splits.items():
        for _ in range(num_scenes):
            scene_id = f"scene{global_scene_idx:04d}_00"
            print(f"Generating {split_name} scene {scene_id}")

            n_obj = np.random.randint(objects_per_scene[0],
                                      objects_per_scene[1] + 1)

            vert, sem, ins, bboxes = build_scene(
                num_objects=n_obj,
                points_per_object=points_per_object,
                scene_extent=scene_extent,
                primitive_types=primitive_types,
            )

            np.save(output_dir / f"{scene_id}_vert.npy", vert)
            np.save(output_dir / f"{scene_id}_sem_label.npy", sem.astype(np.int32))
            np.save(output_dir / f"{scene_id}_ins_label.npy", ins.astype(np.int32))
            np.save(output_dir / f"{scene_id}_bbox.npy", bboxes.astype(np.float32))

            split_files[split_name].write(scene_id + "\n")

            global_scene_idx += 1

    for f in split_files.values():
        f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ScanNet-like dataset from Open3D primitives"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output dataset directory"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=50,
        help="Number of train scenes"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=10,
        help="Number of val scenes"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=10,
        help="Number of test scenes"
    )
    parser.add_argument(
        "--min_objects",
        type=int,
        default=3,
        help="Minimum objects per scene"
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=6,
        help="Maximum objects per scene"
    )
    parser.add_argument(
        "--points_per_object",
        type=int,
        default=2048,
        help="Number of points sampled per object"
    )
    parser.add_argument(
        "--scene_extent",
        type=float,
        default=2.0,
        help="Half-size of scene cube along x and y axes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        objects_per_scene=(args.min_objects, args.max_objects),
        points_per_object=args.points_per_object,
        scene_extent=args.scene_extent,
        primitive_types=list(PRIMITIVE_CLASSES.keys()),
        seed=args.seed,
    )
