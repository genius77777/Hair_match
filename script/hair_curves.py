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
import torch
import copy


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
        vertex_v_t = np.zeros((obj.vs.shape[0], 3))
        vertex_vt_t = np.zeros((obj.vts.shape[0], 3))
        vt_from_v = np.zeros(obj.vts.shape[0], int)
        
        for i in tqdm.tqdm(range(len(obj.fvs)), disable=True):
            # iterate over faces
            vertex_v_n[obj.fvs[i]] += N_face[i]
            vertex_v_t[obj.fvs[i]] += T_face[i]
            vertex_vt_t[obj.fvts[i]] += T_face[i]
            vt_from_v[obj.fvts[i]] = obj.fvs[i]
        
        vertex_v_n = vertex_v_n / np.linalg.norm(vertex_v_n, axis=-1, keepdims=True)
        vertex_v_t = vertex_v_t / np.linalg.norm(vertex_v_t, axis=-1, keepdims=True)

        vertex_vt_n = vertex_v_n[vt_from_v]
        vertex_vt_t = vertex_vt_t / np.linalg.norm(vertex_vt_t, axis=-1, keepdims=True)
        return vertex_vt_n, vertex_vt_t, vertex_v_n, vertex_v_t

    obj = zlw.read_obj(path, tri=True)
    vtn, vtt, vn, vt = compute_NT(obj)
    vtnt = np.stack([vtn, vtt], axis=1)  # [N_vt, 2]
    vnt = np.stack([vn, vt], axis=1)  # [N_vt, 2]
    obj.__setattr__("vtnt", vtnt)
    obj.__setattr__("vnt", vnt)

    return obj


def get_face_transform(neutral_model, target_model, face_id, ratio):
    """
    face_id: [B]
    ratio: [B, 3]
    old_ntb: [B, 3, 3]
    new_ntb: [B, 3, 3]
    """
    B = face_id.shape[0]

    old_nt = (ratio[:, :, None, None] * neutral_model.vtnt[neutral_model.fvts[face_id]]).sum(axis=1)
    old_nt /= np.linalg.norm(old_nt, axis=2, keepdims=True)
    old_ntb = np.concatenate([old_nt, np.cross(old_nt[:, 0], old_nt[:, 1])[:, None]], axis=1).transpose(0, 2, 1)

    new_nt = (ratio[:, :, None, None] * target_model.vtnt[target_model.fvts[face_id]]).sum(axis=1)
    new_nt /= np.linalg.norm(new_nt, axis=2, keepdims=True)
    new_ntb = np.concatenate([new_nt, np.cross(new_nt[:, 0], new_nt[:, 1])[:, None]], axis=1).transpose(0, 2, 1)

    old_roots = (ratio[:, :, None] * neutral_model.vs[neutral_model.fvs[face_id]]).sum(axis=1)
    new_roots = (ratio[:, :, None] * target_model.vs[target_model.fvs[face_id]]).sum(axis=1)

    l2w_old = np.zeros((B, 4, 4))
    l2w_old[:, :3, :3] = old_ntb
    l2w_old[:, :3, 3] = old_roots
    l2w_old[:, 3, 3] = 1

    l2w_new = np.zeros((B, 4, 4))
    l2w_new[:, :3, :3] = new_ntb
    l2w_new[:, :3, 3] = new_roots
    l2w_new[:, 3, 3] = 1

    def fast_inv(RT):
        RT2 = np.zeros_like(RT)
        RT2[:, :3, :3] = RT[:, :3, :3].transpose(0, 2, 1)
        RT2[:, :3, 3:] = RT2[:, :3, :3]@-RT[:, :3, 3:]
        RT2[:, 3, 3] = 1
        return RT2

    w2w = l2w_new@fast_inv(l2w_old)

    return w2w

def get_vert_transform(neutral_model, target_model, vert_id):
    """
    face_id: [B]
    ratio: [B, 3]
    old_ntb: [B, 3, 3]
    new_ntb: [B, 3, 3]
    """
    B = vert_id.shape[0]

    old_nt = neutral_model.vnt[vert_id]
    old_nt /= np.linalg.norm(old_nt, axis=2, keepdims=True)
    old_ntb = np.concatenate([old_nt, np.cross(old_nt[:, 0], old_nt[:, 1])[:, None]], axis=1).transpose(0, 2, 1)

    new_nt = target_model.vnt[vert_id]
    new_nt /= np.linalg.norm(new_nt, axis=2, keepdims=True)
    new_ntb = np.concatenate([new_nt, np.cross(new_nt[:, 0], new_nt[:, 1])[:, None]], axis=1).transpose(0, 2, 1)

    old_roots = neutral_model.vs[vert_id]
    new_roots = target_model.vs[vert_id]

    l2w_old = np.zeros((B, 4, 4))
    l2w_old[:, :3, :3] = old_ntb
    l2w_old[:, :3, 3] = old_roots
    l2w_old[:, 3, 3] = 1

    l2w_new = np.zeros((B, 4, 4))
    l2w_new[:, :3, :3] = new_ntb
    l2w_new[:, :3, 3] = new_roots
    l2w_new[:, 3, 3] = 1

    def fast_inv(RT):
        RT2 = np.zeros_like(RT)
        RT2[:, :3, :3] = RT[:, :3, :3].transpose(0, 2, 1)
        RT2[:, :3, 3:] = RT2[:, :3, :3]@-RT[:, :3, 3:]
        RT2[:, 3, 3] = 1
        return RT2
    w2w = l2w_new@fast_inv(l2w_old)
    return w2w

def transform(neutral_model_path,
              target_model_vs,
              abc_paths,
              output_paths
              ):
    # abc_path = fr"../../All_hair_abc"
    # output_path = fr"../Output_abc_knot"
    # model_path = fr"../../model"
    tmp_dir = os.path.dirname(os.path.abspath(__file__))
    if type(abc_paths) != list():
        abc_paths = [abc_paths]
    if type(output_paths) != list():
        output_paths = [output_paths]
    root_path = fr"{tmp_dir}/../hair_data/hair_curves_roots"
    knot_path = fr'{tmp_dir}/../hair_data/knots_curves.txt'
    knot_id = np.loadtxt(knot_path,int).tolist()
    knot_id = np.array(knot_id)

    neutral_model = read_obj_ntb(neutral_model_path)
    target_model = copy.deepcopy(neutral_model)
    target_model_vs_align, _, _, _ = zlw.align_vertices(target_model_vs, neutral_model.vs, scale=True)
    _, R_trans, T_trans, _ = zlw.align_vertices(target_model_vs_align, target_model_vs, scale=True)
    target_model.vs = target_model_vs_align
    for id_index, abc_path in enumerate(abc_paths):
        identity = os.path.basename(abc_path)[:-4]
        iarch = IArchive(abc_path)
        iarch_top = iarch.getTop()
        objects = iarch_top.children
        oarch = OArchive(output_paths[id_index])
        oarch_top = oarch.getTop()
        hair_num = 0
        for index, object in enumerate(objects):

            i_hair_trans = IXform(object, WrapExistingFlag.kWrapExisting)
            o_hair_trans = OXform(oarch_top, i_hair_trans.getName())
            # assert i_hair_trans.getNumChildren() == 1
            for child_id in range(i_hair_trans.getNumChildren()):
                i_hair_data = ICurves(i_hair_trans.children[child_id], WrapExistingFlag.kWrapExisting)
                i_hair_data_sample = i_hair_data.getSchema().getValue()
                positions = i_hair_data_sample.getPositions()
                positions_np = arrayToNumpy(positions)
                # hair_name = hair_names[index]
                hair_name = object.getName().replace(':',"_")
                roots_data = np.load(fr"{root_path}/{identity}/{hair_name}_grp{child_id}.npy", allow_pickle=True).item()
                
                # roots_data["weight"] = ((roots_data["weight"]**0.25)*255).astype(np.uint8)
                # roots_data["weight"] = (roots_data["weight"].astype(np.float32)/255)**4
                roots_data["ratio"] = roots_data["ratio"].astype(np.float32)/255
                roots_data["weight"] = (roots_data["weight"].astype(np.float32)/255)**4
                roots_data["weight2knot"] = roots_data["weight2knot"].astype(np.float32)/255
                
                hair_num += i_hair_data_sample.getNumCurves()

                positions_np_homo = np.concatenate([positions_np, np.ones((positions_np.shape[0], 1))], axis=1)
                positions_np_homo_batch = positions_np_homo.reshape(i_hair_data_sample.getNumCurves(), -1, 4, 1)

                RT = get_face_transform(neutral_model, target_model, roots_data["face_id"], roots_data["ratio"])
                RT_gpu = torch.tensor(RT[:, None], dtype = torch.float16).to(device = 'cuda')
                positions_np_homo_batch_gpu = torch.tensor(positions_np_homo_batch, dtype = torch.float16).to(device = 'cuda')
                positions_np_homo_batch_transformed_gpu = RT_gpu @ positions_np_homo_batch_gpu
                root_position_np_gpu = positions_np_homo_batch_transformed_gpu[:, :, :3, 0].reshape(-1,3)

                RT = get_vert_transform(neutral_model, target_model, knot_id)
                batch_size = 10000
                knot_position_np_gpu = []
                positions_np_homo = positions_np_homo_batch.reshape(-1, 4, 1)
                for i in range(0, positions_np_homo.shape[0], batch_size):
                    RT_gpu = torch.tensor(RT[:, None], dtype = torch.float16).to(device = 'cuda')
                    positions_np_homo_gpu = torch.tensor(positions_np_homo[None, i:min(i+batch_size,positions_np_homo.shape[0]), :], dtype = torch.float16).to(device='cuda')
                    
                    positions_np_homo_transformed_gpu = RT_gpu @ positions_np_homo_gpu
                    all_knot_position_np_gpu = positions_np_homo_transformed_gpu[:, :, :3, 0]
                    
                    dist_weight_gpu = torch.tensor(roots_data['weight'].transpose()[:, i:min(i+batch_size,positions_np_homo.shape[0]), None], dtype = torch.float16).to(device='cuda')

                    tmp_knot_position_np_gpu = (dist_weight_gpu * all_knot_position_np_gpu).sum(axis=0)
                    knot_position_np_gpu.append(tmp_knot_position_np_gpu)
                
                knot_position_np_gpu = torch.cat(knot_position_np_gpu, dim = 0)
                

                weight2knot_gpu = torch.tensor(roots_data['weight2knot'], dtype = torch.float16).to(device = 'cuda')

                positions_np[:] = (R_trans @ (root_position_np_gpu * (1 - weight2knot_gpu) + knot_position_np_gpu * weight2knot_gpu).cpu().numpy().T + T_trans).T
                # positions_np[:] = (root_position_np_gpu * (1 - weight2knot_gpu) + knot_position_np_gpu * weight2knot_gpu).cpu().numpy()

                o_hair_data_sample = OCurvesSchemaSample()
                o_hair_data_sample.setPositions(positions)
                o_hair_data_sample.setCurvesNumVertices(i_hair_data_sample.getCurvesNumVertices())
                o_hair_data_sample.setType(i_hair_data_sample.getType())
                o_hair_data_sample.setWrap(i_hair_data_sample.getWrap())
                o_hair_data_sample.setBasis(i_hair_data_sample.getBasis())
                if i_hair_data_sample.getOrders() is not None:
                    o_hair_data_sample.setOrders(i_hair_data_sample.getOrders())
                o_hair_data_sample.setKnots(i_hair_data_sample.getKnots())
                o_hair_data = OCurves(o_hair_trans, i_hair_data.getName())
                o_hair_data.getSchema().set(o_hair_data_sample)
