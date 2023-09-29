from imath import *
from alembic.AbcCoreAbstract import *
from alembic.Abc import *
from alembic.AbcGeom import *
from icecream import ic
import numpy as np
import time
from imathnumpy import *
import os
import zlw
import tqdm
import trimesh


def read_obj_ntb(path):

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
    n, t = compute_NT(obj)
    nt = np.stack([n, t], axis=1)  # [N_vt, 2]
    obj.__setattr__("nt", nt)

    return obj


abc_path = fr"./hair_data/hair_curves"
root_dir = fr'./hair_data/hair_curves_roots'
model_path = fr"./head_model"

neutral_model = read_obj_ntb(fr"{model_path}/neutral_model.obj")

tri_mesh = trimesh.Trimesh(vertices=neutral_model.vs, faces=neutral_model.fvs)
tri_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(tri_mesh)


for id_name in os.listdir(os.path.join(abc_path)):

    # if id_name != 'woman_bun1.1.abc':
    #     continue

    knot_path = fr'./hair_data/knots_curves.txt'

    iarch = IArchive(fr"{abc_path}/{id_name}")
    iarch_top = iarch.getTop()
    objects = iarch_top.children

    root_path = os.path.join(root_dir, id_name.replace('.abc', ''))
    os.makedirs(root_path, exist_ok=True)

    for index, object in enumerate(objects):
        print(index)
        i_hair_trans = IXform(object, WrapExistingFlag.kWrapExisting)
        print(i_hair_trans.getNumChildren())
        for child_id in range(i_hair_trans.getNumChildren()):
            i_hair_data = ICurves(i_hair_trans.children[child_id], WrapExistingFlag.kWrapExisting)
            i_hair_data_sample = i_hair_data.getSchema().getValue()
            positions = i_hair_data_sample.getPositions()
            positions_np = arrayToNumpy(positions)

            print(object.getName().replace(':', "_"))
            # ic(positions_np.shape)
            hair_roots = {}
            hair_roots['face_id'] = []
            hair_roots['ratio'] = []

            tmp_positions_np = positions_np.reshape(i_hair_data_sample.getNumCurves(), -1, 3)
            tmp_positions_np = tmp_positions_np[:, 0]
            roots, dists, face_ids = trimesh.proximity.closest_point(tri_mesh, tmp_positions_np)
            ratios = trimesh.triangles.points_to_barycentric(neutral_model.vs[neutral_model.fvs[face_ids]], roots)
            hair_roots['face_id'] = face_ids
            hair_roots['ratio'] = ratios

            knot_id = np.loadtxt(knot_path, int).tolist()
            centers = neutral_model.vs[knot_id]
            # centers = neutral_model.vs[neutral_model.fvs[knot_id]].sum(axis=1)/3
            dists = np.linalg.norm(positions_np[:, None, :] - centers[None, :, :], axis=2)
            dists = np.exp(-(dists*dists)/20)
            weight = dists / dists.sum(axis=1, keepdims=True)
            hair_roots['weight'] = weight

            divide_positions_np = positions_np.reshape(i_hair_data_sample.getNumCurves(), -1, 3)
            dist2root = np.linalg.norm(divide_positions_np - roots[:, None, :], axis=2)
            hair_roots['weight2knot'] = (dist2root/5).reshape(-1, 1)
            hair_roots['weight2knot'] = hair_roots['weight2knot'].clip(min=0, max=1)

            hair_roots['ratio'] = (hair_roots['ratio'] * 255).astype(np.uint8)
            hair_roots['weight'] = ((hair_roots['weight'] ** 0.25) * 255).astype(np.uint8)
            hair_roots['weight2knot'] = (hair_roots['weight2knot'] * 255).astype(np.uint8)
            # ((hair_len - dist2root) / hair_len).reshape(-1,1)

            np.save(os.path.join(root_path, object.getName().replace(':', "_")+fr"_grp{child_id}.npy"), hair_roots)
