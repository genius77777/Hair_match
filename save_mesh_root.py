from icecream import ic
import trimesh
import numpy as np
import time
import os
import zlw
import tqdm

def find_fa(father_dict, v):
    root = v
    while father_dict[root] != -1:
        root = father_dict[root]
    while v != root:
        fa = father_dict[v]
        father_dict[v] = root
        v = fa
    return root

def read_obj_nt(path, if_nt):

    def compute_NT(obj):
        face_pos = obj.vs[obj.fvs.reshape((-1,))].reshape((-1, 3, 3))
        face_uv = obj.vts[obj.fvts.reshape((-1,))].reshape((-1, 3, 2))
        dpos1 = face_pos[:, 1] - face_pos[:, 0]
        dpos2 = face_pos[:, 2] - face_pos[:, 0]
        duv1 = face_uv[:, 1] - face_uv[:, 0]
        duv2 = face_uv[:, 2] - face_uv[:, 0]
        N_face = np.cross(face_pos[:, 0] - face_pos[:, 1], face_pos[:, 1] - face_pos[:, 2])
        N_face /= np.linalg.norm(N_face, axis=-1)[:, None]
        T_face = (dpos1 * duv2[:, 1:2] - dpos2 * duv1[:, 1:2]) / (duv1[:, 0:1] * duv2[:, 1:2] - duv2[:, 0:1] * duv1[:, 1:2])
        T_face /= np.linalg.norm(T_face, axis=-1)[:, None]

        vertex_v_n = np.zeros((obj.vs.shape[0], 3))
        vertex_vt_t = np.zeros((obj.vts.shape[0], 3))
        vt_from_v = np.zeros(obj.vts.shape[0], int)
        with zlw.Timer("assign"):
            for i in tqdm.tqdm(range(len(obj.fvs)), disable=True):
                # iterate over faces
                vertex_v_n[obj.fvs[i]] += N_face[i]
                vertex_vt_t[obj.fvts[i]] += T_face[i]
                vt_from_v[obj.fvts[i]] = obj.fvs[i]
        vertex_v_n = vertex_v_n / np.linalg.norm(vertex_v_n, axis=-1, keepdims=True)
        vertex_vt_n = vertex_v_n[vt_from_v]
        vertex_vt_t = vertex_vt_t / np.linalg.norm(vertex_vt_t, axis=-1, keepdims=True)
        return vertex_vt_n, vertex_vt_t

    obj = zlw.read_obj(path, tri=True)
    if if_nt:
        n, t = compute_NT(obj)
        nt = np.stack([n, t], axis=1)  # [N_vt, 2]
        obj.__setattr__("nt", nt)

    return obj

head_model_path = fr"./head_model"
hair_model_dir = fr"./hair_data/hair_mesh"
root_save_dir = fr"./hair_data/hair_mesh_roots"
knot_path = fr'./hair_data/knots_mesh.txt'
knot_id = np.loadtxt(knot_path, int).tolist()
os.makedirs(root_save_dir, exist_ok=True)

with zlw.Timer("read head obj total"):
    neutral_model = read_obj_nt(fr"{head_model_path}/neutral_model.obj", if_nt = True)
    # start_time = time.time()
    # target_model = read_obj_nt(fr"{model_path}/target_model.obj", if_nt = True)
    # end_time = time.time()
    # print("load_target_obj_time(s):", end_time - start_time)

tri_mesh = trimesh.Trimesh(vertices = neutral_model.vs, faces = neutral_model.fvs)
tri_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(tri_mesh)

for hair_model_path in os.listdir(hair_model_dir):

    if hair_model_path[-4:] != ".obj":
        continue

    print(fr"{hair_model_dir}/{hair_model_path}")
    with zlw.Timer("read hair obj total"):
        hair_model = read_obj_nt(fr"{hair_model_dir}/{hair_model_path}", if_nt = False)

    print(hair_model.vs.shape)
    father_dict = np.zeros(hair_model.vs.shape[0]).astype(np.int32) - 1

    for fv in hair_model.fvs:
        fa0 = find_fa(father_dict, fv[0])
        fa1 = find_fa(father_dict, fv[1])
        fa2 = find_fa(father_dict, fv[2])
        if fa0 != fa1:
            father_dict[fa1] = fa0
        if fa0 != fa2:
            father_dict[fa2] = fa0

    roots, dists, face_ids = trimesh.proximity.closest_point(tri_mesh, hair_model.vs)
    ratios = trimesh.triangles.points_to_barycentric(neutral_model.vs[neutral_model.fvs[face_ids]], roots)

    # nearface_node = {}

    # for index in range(hair_model.vs.shape[0]):
    #     tmp_fa = find_fa(father_dict, index)
    #     if tmp_fa not in nearface_node.keys():
    #         nearface_node[tmp_fa] = index
    #     else:
    #         if dists[index] < dists[nearface_node[tmp_fa]]:
    #             nearface_node[tmp_fa] = index

    # hair_roots = {}
    # hair_roots['face_id'] = []
    # hair_roots['ratio'] = []

    # dist2root = []
    # for index in range(hair_model.vs.shape[0]):
    #     tmp_fa = find_fa(father_dict, index)
    #     hair_roots['face_id'].append(face_ids[nearface_node[tmp_fa]])
    #     hair_roots['ratio'].append(ratios[nearface_node[tmp_fa]])
    #     dist2root.append(np.linalg.norm(hair_model.vs[index]-roots[nearface_node[tmp_fa]], axis=0))

    hair_roots = {}
    hair_roots['face_id'] = []
    hair_roots['ratio'] = []
    dist2root = []
    for index in range(hair_model.vs.shape[0]):
        tmp_fa = find_fa(father_dict, index)
        hair_roots['face_id'].append(face_ids[index])
        hair_roots['ratio'].append(ratios[index])
        dist2root.append(np.linalg.norm(hair_model.vs[index]-roots[index], axis=0))


    hair_roots['face_id'] = np.array(hair_roots['face_id'])
    hair_roots['ratio'] = np.array(hair_roots['ratio'])
    dist2root = np.array(dist2root)

    # print(dist2root.shape,dist2root.max(),dist2root.min())

    centers = neutral_model.vs[knot_id]
    # print(hair_model.vs.shape, centers.shape)
    dists = np.linalg.norm(hair_model.vs[:, None, :] - centers[None, :, :], axis=2)
    dists = np.exp(-(dists*dists)/20)
    weight = dists / dists.sum(axis=1, keepdims=True)
    hair_roots['weight'] = weight


    hair_roots['weight2knot'] = (dist2root/5)
    hair_roots['weight2knot'] = hair_roots['weight2knot'].clip(min=0, max=1)

    hair_roots['ratio'] = (hair_roots['ratio'] * 255).astype(np.uint8)
    hair_roots['weight'] = ((hair_roots['weight'] ** 0.25) * 255).astype(np.uint8)
    hair_roots['weight2knot'] = (hair_roots['weight2knot'] * 255).astype(np.uint8)

    np.save(fr"{root_save_dir}/{hair_model_path[:-4]}.npy", hair_roots)
    # break
