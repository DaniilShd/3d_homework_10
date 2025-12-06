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
                     size_range=(0.1, 0.8),
                     height_range=(0.2, 1.0),
                     aspect_ratio_range=(0.5, 2.0)):
    s_min, s_max = size_range
    h_min, h_max = height_range
    ar_min, ar_max = aspect_ratio_range

    if primitive_type == "box":
        aspect_ratio = np.random.uniform(ar_min, ar_max)
        base_size = np.random.uniform(s_min, s_max)
        sx = base_size
        sy = base_size * aspect_ratio if np.random.rand() > 0.5 else base_size
        sz = base_size * np.random.uniform(0.5, 1.5)
        mesh = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)

    elif primitive_type == "sphere":
        r = np.random.uniform(s_min * 0.5, s_max * 0.5)
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=30)

    elif primitive_type == "cylinder":
        r = np.random.uniform(s_min * 0.3, s_max * 0.5)
        h = np.random.uniform(h_min, h_max)
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h, resolution=30)

    elif primitive_type == "torus":
        r_torus = np.random.uniform(s_min * 0.4, s_max * 0.6)
        r_tube = r_torus * np.random.uniform(0.1, 0.4)
        mesh = o3d.geometry.TriangleMesh.create_torus(
            torus_radius=r_torus,
            tube_radius=r_tube,
            radial_resolution=30,
            tubular_resolution=20
        )

    elif primitive_type == "cone":
        r = np.random.uniform(s_min * 0.3, s_max * 0.5)
        h = np.random.uniform(h_min, h_max)
        mesh = o3d.geometry.TriangleMesh.create_cone(radius=r, height=h, resolution=30)

    else:
        raise ValueError(f"Unknown primitive type: {primitive_type}")

    mesh.compute_vertex_normals()
    
    # Более разнообразные цвета с лучшей различимостью
    color_scheme = np.random.choice(['pastel', 'vibrant', 'dark'])
    if color_scheme == 'pastel':
        color = np.random.uniform(0.6, 0.95, size=3)
    elif color_scheme == 'vibrant':
        color = np.random.uniform(0.2, 0.9, size=3)
        color[np.random.randint(0, 3)] = np.random.uniform(0.8, 1.0)
    else:  # dark
        color = np.random.uniform(0.1, 0.5, size=3)
    
    mesh.paint_uniform_color(color)

    return mesh


def random_transform(mesh,
                     scene_extent=2.0,
                     max_rotation_deg=180.0):
    # Позиция с учетом предотвращения сильного пересечения
    tx = np.random.uniform(-scene_extent * 0.8, scene_extent * 0.8)
    ty = np.random.uniform(-scene_extent * 0.8, scene_extent * 0.8)
    tz = np.random.uniform(-scene_extent * 0.1, scene_extent * 0.1)

    # Полноценное 3D вращение
    angles = np.deg2rad(np.random.uniform(-max_rotation_deg, max_rotation_deg, size=3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    
    R = Rx @ Ry @ Rz

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz])

    mesh_t = mesh.transform(T)
    return mesh_t, T


def sample_points_from_mesh(mesh,
                            num_points=4096,
                            add_noise=True):
    # Используем равномерную или пуассоновскую дискретизацию для разнообразия
    if np.random.rand() > 0.5:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    else:
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    
    # Добавляем реалистичный шум
    if add_noise:
        points = np.asarray(pcd.points)
        noise = np.random.normal(0, 0.005, points.shape)  # Меньший шум для четкости
        points += noise
        pcd.points = o3d.utility.Vector3dVector(points)
    
    if not pcd.has_normals():
        pcd.estimate_normals()
    
    return pcd


def build_scene(num_objects,
                points_per_object,
                scene_extent,
                primitive_types,
                min_distance=0.3):
    all_points = []
    all_colors = []
    all_normals = []
    all_semantic = []
    all_instance = []
    bboxes = []
    
    object_positions = []
    object_sizes = []

    for inst_id in range(num_objects):
        attempts = 0
        placed = False
        
        while not placed and attempts < 20:
            primitive_type = np.random.choice(primitive_types)
            mesh = create_primitive(primitive_type)
            mesh_t, transform = random_transform(mesh, scene_extent=scene_extent)
            
            # Получаем AABB для проверки коллизий
            aabb = mesh_t.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extent = aabb.get_extent()
            size = np.max(extent)
            
            # Проверяем расстояние до других объектов
            collision = False
            for pos, other_size in zip(object_positions, object_sizes):
                distance = np.linalg.norm(center[:2] - pos[:2])  # Только XY
                if distance < (size + other_size) * 0.5 * min_distance:
                    collision = True
                    break
            
            if not collision:
                placed = True
                object_positions.append(center)
                object_sizes.append(size)
                
                pcd = sample_points_from_mesh(mesh_t, num_points=points_per_object)
                
                pts = np.asarray(pcd.points, dtype=np.float32)
                cols = np.asarray(pcd.colors, dtype=np.float32)
                normals = np.asarray(pcd.normals, dtype=np.float32)
                
                all_points.append(pts)
                all_colors.append(cols)
                all_normals.append(normals)
                
                class_id = PRIMITIVE_CLASSES[primitive_type]
                semantic_labels = np.full(pts.shape[0], class_id, dtype=np.int32)
                instance_labels = np.full(pts.shape[0], inst_id, dtype=np.int32)
                
                all_semantic.append(semantic_labels)
                all_instance.append(instance_labels)
                
                # Более точный bounding box с ориентацией
                obb = mesh_t.get_oriented_bounding_box()
                bbox_center = obb.center
                bbox_extent = obb.extent
                bbox_R = obb.R
                
                # Сохраняем как 9 параметров: центр(3), размер(3), ориентация(3)
                bbox = np.zeros(10, dtype=np.float32)
                bbox[0:3] = bbox_center
                bbox[3:6] = bbox_extent
                bbox[6:9] = bbox_R.flatten()[:3]  # Первая колонка матрицы вращения
                bbox[9] = class_id
                bboxes.append(bbox)
            
            attempts += 1
    
    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_normals = np.concatenate(all_normals, axis=0)
        all_semantic = np.concatenate(all_semantic, axis=0)
        all_instance = np.concatenate(all_instance, axis=0)
        bboxes = np.stack(bboxes, axis=0)
        
        # Нормализуем точки для лучшей стабильности обучения
        points_mean = np.mean(all_points, axis=0)
        all_points -= points_mean
        for i in range(len(bboxes)):
            bboxes[i, 0:3] -= points_mean
    else:
        all_points = np.zeros((0, 3), dtype=np.float32)
        all_colors = np.zeros((0, 3), dtype=np.float32)
        all_normals = np.zeros((0, 3), dtype=np.float32)
        all_semantic = np.zeros((0,), dtype=np.int32)
        all_instance = np.zeros((0,), dtype=np.int32)
        bboxes = np.zeros((0, 10), dtype=np.float32)

    # Собираем все признаки: точки, цвета, нормали
    vert = np.concatenate([all_points, all_colors, all_normals], axis=1)

    return vert, all_semantic, all_instance, bboxes


def generate_dataset(output_dir,
                     num_train=200,  # Увеличено для лучшего обучения
                     num_val=30,
                     num_test=30,
                     objects_per_scene=(2, 10),  # Более широкий диапазон
                     points_per_object=2048,
                     scene_extent=3.0,  # Увеличена сцена
                     primitive_types=None,
                     seed=42):
    if primitive_types is None:
        primitive_types = list(PRIMITIVE_CLASSES.keys())

    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем метаданные о классах
    with open(output_dir / "classes.txt", "w") as f:
        for name, idx in PRIMITIVE_CLASSES.items():
            f.write(f"{idx} {name}\n")

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
            
            # Для тренировочных сцен иногда создаем более сложные/простые сцены
            if split_name == "train":
                if np.random.rand() < 0.2:
                    n_obj = objects_per_scene[0]  # Простая сцена
                elif np.random.rand() < 0.2:
                    n_obj = objects_per_scene[1]  # Сложная сцена

            vert, sem, ins, bboxes = build_scene(
                num_objects=n_obj,
                points_per_object=points_per_object,
                scene_extent=scene_extent,
                primitive_types=primitive_types,
            )

            # Балансировка классов (для тренировочных данных)
            if split_name == "train":
                unique_classes = np.unique(sem)
                # Убедимся, что в сцене есть минимум 2 разных класса
                if len(unique_classes) < 2 and n_obj > 1:
                    # Перегенерируем сцену с гарантией разных классов
                    attempts = 0
                    while len(unique_classes) < 2 and attempts < 10:
                        vert, sem, ins, bboxes = build_scene(
                            num_objects=n_obj,
                            points_per_object=points_per_object,
                            scene_extent=scene_extent,
                            primitive_types=primitive_types,
                        )
                        unique_classes = np.unique(sem)
                        attempts += 1

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
        default=200,  # Увеличено
        help="Number of train scenes"
    )
    parser.add_argument(
        "--num_val",
        type=int,
        default=30,
        help="Number of val scenes"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=30,
        help="Number of test scenes"
    )
    parser.add_argument(
        "--min_objects",
        type=int,
        default=2,  # Уменьшено
        help="Minimum objects per scene"
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=10,  # Увеличено
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
        default=3.0,  # Увеличено
        help="Half-size of scene cube along x and y axes"
    )
    parser.add_argument(
        "--add_normals",
        action="store_true",
        default=True,
        help="Add normal vectors to point cloud features"
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