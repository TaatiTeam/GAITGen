import os

import clip
import numpy as np
import torch
import copy
from os.path import join as pjoin
# from scipy import linalg
from utils.metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from utils.utils import DEVICE
from utils.plot_script import plot_3d_motion
import wandb
from collections import defaultdict
from utils.paramUtil import t2m_kinematic_chain
from data.PD_Unified_dataloader import get_data_loaders


def plott(joints, caption, result_dir, last_folder, kinematic_chain):
    joints = np.asarray(joints)
    if joints.ndim == 4 and joints.shape[0] == 1:
        joints = np.squeeze(joints, axis=0)
    save_dir = pjoin(result_dir, last_folder)
    os.makedirs(save_dir, exist_ok=True)
    plot_3d_motion(pjoin(save_dir, caption + '.mp4'), kinematic_chain, joints, title=caption, fps=20)
#
#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)

def get_sample_by_class(dataloader, class_label, num_samples=1):
    samples = []
    for batch in dataloader:
        motions, severity_labels, m_lengths = batch
        # Filter the samples that belong to the specific class
        for motion, severity_label, m_length in zip(motions, severity_labels, m_lengths):
            if severity_label.item() == class_label:
                samples.append((motion.unsqueeze(0), severity_label, m_length))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    return samples

@torch.no_grad()
def mix_match_eval(opt, net, joints_num, eval_dir):
    mopt = copy.deepcopy(opt)
    mopt.get_whole_motion = True
    mopt.is_train = False
    mopt.batch_size = 256
    mval_loader, val_dataset = get_data_loaders(mopt, split='test')
    
    net.eval()
    class_label1 = 0
    class_label2 = 3
    sample_1 = get_sample_by_class(mval_loader, class_label=class_label1, num_samples=1)[0]
    sample_2 = get_sample_by_class(mval_loader, class_label=class_label2, num_samples=1)[0]
    with torch.no_grad():
        motion_1, severity_1, len_1 = sample_1
        motion_2, severity_2, len_2 = sample_2
        target_len = torch.min(len_1, len_2)
        motion_1 = motion_1[:, :target_len]
        motion_2 = motion_2[:, :target_len]
        
        # Extract motion and disease representations from both samples
        severity_1 = severity_1.unsqueeze(0).to(opt.device)
        severity_2 = severity_2.unsqueeze(0).to(opt.device)
        zero_severity = torch.zeros_like(severity_2).to(opt.device)
        m_code_idx_1, d_code_idx_1, m_quantized_all_1, d_quantized_all_1, m_quantized_1, d_quantized_1 = net.encode(motion_1.to(DEVICE), severity_1.to(DEVICE))
        m_code_idx_2, d_code_idx_2, m_quantized_all_2, d_quantized_all_2, m_quantized_2, d_quantized_2 = net.encode(motion_2.to(DEVICE), severity_2.to(DEVICE))
        
        rec_m1_d2 = net.forward_decoder_MM(m_code_idx_1, d_code_idx_2, severity_2) # use severity of intened diease
        rec_m1_d2 = mval_loader.dataset.inv_transform(rec_m1_d2.detach().cpu().numpy())
        joint_m1_d2 = recover_from_ric(torch.from_numpy(rec_m1_d2).float(), joints_num).numpy()
        
        rec_m1_m1 = net.forward_decoder_MM(m_code_idx_1, m_code_idx_1, zero_severity, ctype='mm')
        rec_m1_m1 = mval_loader.dataset.inv_transform(rec_m1_m1.detach().cpu().numpy())
        joint_m1_m1 = recover_from_ric(torch.from_numpy(rec_m1_m1).float(), joints_num).numpy()
        
        rec_d1_d1 = net.forward_decoder_MM(d_code_idx_1, d_code_idx_1, severity_1, ctype='dd')
        rec_d1_d1 = mval_loader.dataset.inv_transform(rec_d1_d1.detach().cpu().numpy())
        joint_d1_d1 = recover_from_ric(torch.from_numpy(rec_d1_d1).float(), joints_num).numpy()
        
        rec_m1_d1 = net.forward_decoder_MM(m_code_idx_1, d_code_idx_1, severity_1)
        rec_m1_d1 = mval_loader.dataset.inv_transform(rec_m1_d1.detach().cpu().numpy())
        joint_m1_d1 = recover_from_ric(torch.from_numpy(rec_m1_d1).float(), joints_num).numpy()
        
        rec_m2_d1 = net.forward_decoder_MM(m_code_idx_2, d_code_idx_1, severity_1)
        rec_m2_d1 = mval_loader.dataset.inv_transform(rec_m2_d1.detach().cpu().numpy())
        joint_m2_d1 = recover_from_ric(torch.from_numpy(rec_m2_d1).float(), joints_num).numpy()
        
        rec_m2_m2 = net.forward_decoder_MM(m_code_idx_2, m_code_idx_2, zero_severity, ctype='mm')
        rec_m2_m2 = mval_loader.dataset.inv_transform(rec_m2_m2.detach().cpu().numpy())
        joint_m2_m2 = recover_from_ric(torch.from_numpy(rec_m2_m2).float(), joints_num).numpy()
        
        rec_d2_d2 = net.forward_decoder_MM(d_code_idx_2, d_code_idx_2, severity_2, ctype='dd')
        rec_d2_d2 = mval_loader.dataset.inv_transform(rec_d2_d2.detach().cpu().numpy())
        joint_d2_d2 = recover_from_ric(torch.from_numpy(rec_d2_d2).float(), joints_num).numpy()
        
        rec_m2_d2 = net.forward_decoder_MM(m_code_idx_2, d_code_idx_2, severity_2)
        rec_m2_d2 = mval_loader.dataset.inv_transform(rec_m2_d2.detach().cpu().numpy())
        joint_m2_d2 = recover_from_ric(torch.from_numpy(rec_m2_d2).float(), joints_num).numpy()
        
        os.makedirs(pjoin(eval_dir, 'mix_match'), exist_ok=True)
        kinematic_chain = t2m_kinematic_chain
        caption = "sample_motion%d_disease%d"%(class_label1, class_label2)
        plott(joint_m1_d2, caption, eval_dir, 'mix_match', kinematic_chain)
        caption = "sample_motion%d_motion%d"%(class_label1, class_label1)
        plott(joint_m1_m1, caption, eval_dir, 'mix_match', kinematic_chain)
        caption = "sample_disease%d_disease%d"%(class_label1, class_label1)
        plott(joint_d1_d1, caption, eval_dir, 'mix_match', kinematic_chain)
        caption = "sample_motion%d_disease%d"%(class_label1, class_label1)
        plott(joint_m1_d1, caption, eval_dir, 'mix_match', kinematic_chain)
        
        caption = "sample_motion%d_disease%d"%(class_label2, class_label1)
        plott(joint_m2_d1, caption, eval_dir, 'mix_match', kinematic_chain)
        caption = "sample_motion%d_motion%d"%(class_label2, class_label2)
        plott(joint_m2_m2, caption, eval_dir, 'mix_match', kinematic_chain)
        caption = "sample_disease%d_disease%d"%(class_label2, class_label2)
        plott(joint_d2_d2, caption, eval_dir, 'mix_match', kinematic_chain)
        caption = "sample_motion%d_disease%d"%(class_label2, class_label2)
        plott(joint_m2_d2, caption, eval_dir, 'mix_match', kinematic_chain)
        
        # GT
        joint_data_motion_1 = mval_loader.dataset.inv_transform(motion_1)
        joint_data_motion_1 = recover_from_ric(joint_data_motion_1.float(), opt.joints_num).numpy()
        caption = "sample_original_class%d"%(class_label1)
        plott(joint_data_motion_1, caption, eval_dir, 'mix_match', kinematic_chain)
        joint_data_motion_2 = mval_loader.dataset.inv_transform(motion_2)
        joint_data_motion_2 = recover_from_ric(joint_data_motion_2.float(), opt.joints_num).numpy()
        caption = "sample_original_class%d"%(class_label2)
        plott(joint_data_motion_2, caption, eval_dir, 'mix_match', kinematic_chain)
        

@torch.no_grad()
def evaluation_vqvae(opt, out_dir, val_loader, net, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe_mld = 0
    pampjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, label = batch

        motion = motion.to(DEVICE)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        if opt.conditional:
            input = (motion, label.to(DEVICE))
        else:
            input = (motion,)

        pred_pose_eval, _, _, _, _, _,= net(*input)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), opt.joints_num)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), opt.joints_num)
            mpjpe_mld += torch.sum(calc_mpjpe(pred, gt, align_inds=[0]))
            pampjpe += torch.sum(calc_pampjpe(pred, gt))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]
        
        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    factor = 1000
    mpjpe_mld = mpjpe_mld / num_poses * factor
    pampjpe = pampjpe / num_poses * factor

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    # print(msg)

    if draw:
        wandb.log({
        "Test/FID": fid,
        "Test/Diversity": diversity,
        "Test/Diversity_real": diversity_real,
        "Test/top1": R_precision[0],
        "Test/top2": R_precision[1],
        "Test/top3": R_precision[2],
        "Test/matching_score": matching_score_pred,
        "Test/matching_score_real": matching_score_real,
        "Test/MPJPE": mpjpe_mld,
        "Test/PAMPJPE": pampjpe,
        "epoch": ep
        })

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(opt, val_loader, net, repeat_id, eval_wrapper, num_joint, eval_dir, mm=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    mpjpe_mld = 0
    pampjpe = 0
    accel = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))     
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, label = batch

        motion = motion.to(DEVICE)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if opt.conditional:
            input = (motion, label.to(DEVICE))
        else:
            input = (motion,)
        pred_pose_eval, _, _, _, _, _,= net(*input)        
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            mpjpe_mld += torch.sum(calc_mpjpe(pred, gt, align_inds=[0]))
            pampjpe += torch.sum(calc_pampjpe(pred, gt))
            accel += torch.sum(calc_accel(pred, gt))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses
    factor = 1000
    mpjpe_mld = mpjpe_mld / num_poses * factor
    pampjpe = pampjpe / num_poses * factor
    # accel error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    # n-2 for each sequences
    accel = accel / (num_poses - 2 * nb_sample) * factor 

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f, MPJPE_mld. %.4f, PAMPJPE. %.4f, ACCEL. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe, mpjpe_mld, pampjpe, accel)
    # logger.info(msg)
    print(msg)
    file_path = pjoin(opt.outdir, f'eval_results_{repeat_id}.txt')
    with open(file_path, 'a') as f:  # 'a' means append mode, so it doesn't overwrite the file
        f.write(msg + '\n')
   
    if repeat_id == 0 and mm:
        mix_match_eval(opt, net, num_joint, eval_dir)
    
    return fid, diversity, R_precision, matching_score_pred, mpjpe_mld, pampjpe, accel


@torch.no_grad()
def evaluation_vqvae_plus_mpjpe2(opt, val_loader, net, repeat_id, eval_wrapper, num_joint, eval_dir, mm=True):
    net.eval()

    motion_annotation_list, motion_pred_list = [], []
    R_precision_real, R_precision = 0, 0
    nb_sample = 0
    matching_score_real, matching_score_pred = 0, 0
    mpjpe, mpjpe_mld, pampjpe, accel = 0, 0, 0, 0
    pore_pathology_only_mpjpe = 0
    num_poses = 0
    
    for batch in val_loader:
        # print(len(batch))     
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, label = batch

        motion = motion.to(DEVICE)
        label = label.to(DEVICE)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if opt.conditional:
            input = (motion, label)
        else:
            input = (motion,)
        pred_pose_eval, _, _, _, _, _,= net(*input)        
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
        
        # Pathology-only reconstruction (q_p)
        m_code_idx, d_code_idx, _, _, _, _ = net.encode(*input)
        pred_pose_eval_pathology_only = net.forward_decoder_MM(disease_idx=d_code_idx, motion_idx=d_code_idx, y=label, ctype='dd')
        # et_pred_pathology, em_pred_pathology = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval_pathology_only,
        #                                                   m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        bpred_pathology_only = val_loader.dataset.inv_transform(pred_pose_eval_pathology_only.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            pred_pathology_only = recover_from_ric(torch.from_numpy(bpred_pathology_only[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            mpjpe_mld += torch.sum(calc_mpjpe(pred, gt, align_inds=[0]))
            pampjpe += torch.sum(calc_pampjpe(pred, gt))
            accel += torch.sum(calc_accel(pred, gt))
            pore_pathology_only_mpjpe += torch.sum(calc_mpjpe(pred_pathology_only, gt, align_inds=[0]))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]
    
        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses
    factor = 1000
    mpjpe_mld = mpjpe_mld / num_poses * factor
    pampjpe = pampjpe / num_poses * factor
    # accel error: joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    # n-2 for each sequences
    accel = accel / (num_poses - 2 * nb_sample) * factor 
    
    pore_pathology_only_mpjpe = pore_pathology_only_mpjpe / num_poses
    pore = (pore_pathology_only_mpjpe - mpjpe) / (mpjpe + 1e-8) # Higher PORE indicates better disentanglement

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f, MPJPE_mld. %.4f, PAMPJPE. %.4f, ACCEL. %.4f, PORE. %.4f," % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe, mpjpe_mld, pampjpe, accel, pore)
    # logger.info(msg)
    print(msg)
    # file_path = pjoin(opt.outdir, f'eval_results_{repeat_id}.txt')
    # with open(file_path, 'a') as f:  # 'a' means append mode, so it doesn't overwrite the file
    #     f.write(msg + '\n')
   
    if repeat_id == 0 and mm:
        mix_match_eval(opt, net, num_joint, eval_dir)
    
    return fid, diversity, R_precision, matching_score_pred, mpjpe_mld, pampjpe, accel, pore



@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(DEVICE)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.to(DEVICE)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            # pred_vq_codes = pred_vq_codes - pred_vq_res + all_codes[1:].sum(0)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(rec_motions)        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)

        # motion_codes = motion_codes.permute(0, 2, 1)
        mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        # Collecting generated and real motions based on class labels
        for i in range(bs):
            classwise_generated[label[i].item()].append(pred_motions[i].cpu().numpy())
            classwise_real[label[i].item()].append(pose[i].cpu().numpy())

        nb_sample += bs
        
    # compute diversity for each class
    class_diversities = {}
    for class_label, motions in classwise_generated.items():
        if len(motions) > 0:
            motion_np = np.array(motions)
            class_diversities[class_label] = calculate_diversity(motion_np, min(300, len(motions)-1))
    for class_label, diversity in class_diversities.items():
        wandb.log({f"Test_class/Diversity_Class_{class_label}": diversity, "epoch": ep})

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    wandb.log({
        "Test/FID": fid,
        "Test/Diversity": diversity,
        "Test/top1": R_precision[0],
        "Test/top2": R_precision[1],
        "Test/top3": R_precision[2],
        "Test/matching_score": matching_score_pred,
        "epoch": ep
    })



    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching

@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans, vq_model, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE).long()
        pose = pose.to(DEVICE).float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        if 'Conditional' in vq_model.__class__.__name__:
            input = (pose, label.to(DEVICE))
        else:
            input = (pose,)
        code_indices, all_codes = vq_model.encode(*input) #(bs, T/4, 6), (6, bs, code_dim, T/4)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        # Collecting generated and real motions based on class labels
        for i in range(bs):
            classwise_generated[label[i].item()].append(em_pred[i].cpu().numpy())
            classwise_real[label[i].item()].append(em[i].cpu().numpy())

        nb_sample += bs
    
    # compute diversity for each class
    class_diversities = {}
    for class_label, motions in classwise_generated.items():
        if len(motions) > 0:
            motion_np = np.array(motions)
            class_diversities[class_label] = calculate_diversity(motion_np, min(300, len(motions)-1))
    for class_label, diversity in class_diversities.items():
        wandb.log({f"Test_class/Diversity_Class_{class_label}": diversity, "epoch": ep})

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    wandb.log({"Test/FID": fid, "epoch": ep})
    wandb.log({"Test/Diversity": diversity, "epoch": ep})
    wandb.log({"Test/top1": R_precision[0], "epoch": ep})
    wandb.log({"Test/top2": R_precision[1], "epoch": ep})
    wandb.log({"Test/top3": R_precision[2], "epoch": ep})
    wandb.log({"Test/matching_score": matching_score_pred, "epoch": ep})



    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_res_transformer_plus_l1(val_loader, vq_model, trans, repeat_id, eval_wrapper, num_joint,
                                       cond_scale=2, temperature=1, topkr=0.9, cal_l1=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(DEVICE).long()
        pose = pose.to(DEVICE).float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # print(code_indices[0:2, :, 1])

        pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4, topk_filter_thres=topkr,
                                  temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            for i in range(bs):
                gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
                pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
                # gt = motion[i, :m_length[i]]
                # pred = pred_pose_eval[i, :m_length[i]]
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.to(DEVICE)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                mids.unsqueeze_(-1)
                pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            mids.unsqueeze_(-1)
            pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        
        # Store generated and real motions per class
        for j in range(bs):
            classwise_generated[label[j].item()].append(em_pred[j].cpu().numpy())
            classwise_real[label[j].item()].append(em[j].cpu().numpy())

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_label, real_motions in sorted(classwise_real.items()):
        if len(real_motions) > 0:
            real_np = np.array(real_motions)
            generated_np = np.array(classwise_generated[class_label])

            class_diversity = calculate_diversity(generated_np, min(300, len(generated_np)-1))
            class_diversity_real = calculate_diversity(real_np, min(300, len(real_np)-1))
            class_fid = calculate_frechet_distance(*calculate_activation_statistics(real_np),
                                                   *calculate_activation_statistics(generated_np))
            # class_R_precision = calculate_R_precision(real_np, generated_np, top_k=3, sum_all=True)
            print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity: {class_diversity:.4f} (real: {class_diversity_real:.4f})")#, R_precision: {class_R_precision:.4f}")
            
            per_class_metrics[class_label] = {
                'fid': class_fid,
                'diversity': class_diversity,
                'diversity_real': class_diversity_real,
                # 'R_precision': class_R_precision
            }

    return fid, diversity, R_precision, matching_score_pred, multimodality, per_class_metrics



@torch.no_grad()
def evaluation_Dmask_transformer(out_dir, val_loader, trans, vq_model, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # (b, seqlen)
        _, ids_m, ids_d = trans.generate(clip_text, m_length//4, m_length//4, time_steps, cond_scale, temperature=1)

        # motion_codes = motion_codes.permute(0, 2, 1)
        ids_m.unsqueeze_(-1)
        ids_d.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(ids_m, ids_d, y=label.to(DEVICE))

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        # Collecting generated and real motions based on class labels
        for i in range(bs):
            # classwise_generated[label[i].item()].append(pred_motions[i].cpu().numpy())
            # classwise_real[label[i].item()].append(pose[i].cpu().numpy())
            classwise_generated[label[i].item()].append(em_pred[i].cpu().numpy())
            classwise_real[label[i].item()].append(em[i].cpu().numpy())

        nb_sample += bs
        
    # compute diversity for each class
    class_diversities = {}
    for class_label, motions in classwise_generated.items():
        if len(motions) > 0:
            motion_np = np.array(motions)
            class_diversities[class_label] = calculate_diversity(motion_np, min(300, len(motions)-1))
    for class_label, diversity in class_diversities.items():
        wandb.log({f"Test_class/Diversity_Class_{class_label}": diversity, "epoch": ep})

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    wandb.log({
        "Test/FID": fid,
        "Test/Diversity": diversity,
        "Test/top1": R_precision[0],
        "Test/top2": R_precision[1],
        "Test/top3": R_precision[2],
        "Test/matching_score": matching_score_pred,
        "epoch": ep
    })

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching




@torch.no_grad()
def evaluation_Dres_transformer(out_dir, val_loader, trans, vq_model, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1, vq_conditional=False):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE).long()
        pose = pose.to(DEVICE).float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        
        if vq_conditional:
             inp = (pose, label.to(DEVICE))
        else:
             inp = (pose,)
        
        code_indices_m, code_indices_d, all_codes, _, _, _ = vq_model.encode(*inp) #(bs, T/4, 6), (6, bs, code_dim, T/4)
        # (b, seqlen)
        if ep == 0:
            pred_ids_m = code_indices_m[..., 0:1]
            pred_ids_d = code_indices_d[..., 0:1]
        else:
            pred_ids_combine, pred_ids_m, pred_ids_d = trans.generate(code_indices_m[..., 0], code_indices_d[..., 0], clip_text, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        # Collecting generated and real motions based on class labels
        for i in range(bs):
            classwise_generated[label[i].item()].append(em_pred[i].cpu().numpy())
            classwise_real[label[i].item()].append(em[i].cpu().numpy())

        nb_sample += bs
    
    # compute diversity for each class
    class_diversities = {}
    for class_label, motions in classwise_generated.items():
        if len(motions) > 0:
            motion_np = np.array(motions)
            class_diversities[class_label] = calculate_diversity(motion_np, min(300, len(motions)-1))
    for class_label, diversity in class_diversities.items():
        wandb.log({f"Test_class/Diversity_Class_{class_label}": diversity, "epoch": ep})

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    wandb.log({"Test/FID": fid, "epoch": ep})
    wandb.log({"Test/Diversity": diversity, "epoch": ep})
    wandb.log({"Test/top1": R_precision[0], "epoch": ep})
    wandb.log({"Test/top2": R_precision[1], "epoch": ep})
    wandb.log({"Test/top3": R_precision[2], "epoch": ep})
    wandb.log({"Test/matching_score": matching_score_pred, "epoch": ep})



    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching



@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_disent_all(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, out_path=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()
    saved_file_path = os.path.join(out_path, 'generated_data.pkl') if out_path else None

    motion_annotation_list = []
    motion_pred_list = []
    em_pred_allgen_list =[]
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    all_gt_poses = []
    all_pred_poses = []
    all_gt_mlengths = []
    all_pred_mlengths = []
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        em_pred_allgen = []
        n_gen_samples = 30

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(n_gen_samples):
                _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)

                pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))


                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
                em_pred_allgen.append(em_pred)
                all_pred_poses.append(pred_motions)
                all_pred_mlengths.append(m_length)
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
            em_pred_allgen = torch.cat(em_pred_allgen, dim=0)  # Shape: (bs * n_gen_samples, embedding_dim)
        else:
            for _ in range(n_gen_samples):
                _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                    temperature=temperature, topk_filter_thres=topkr,
                                    force_mask=force_mask)

                _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)

                pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                                pred_motions.clone(),
                                                                m_length)
                em_pred_allgen.append(em_pred)
                all_pred_poses.append(pred_motions)
                all_pred_mlengths.append(m_length)
            em_pred_allgen = torch.cat(em_pred_allgen, dim=0)  # Shape: (bs * n_gen_samples, embedding_dim)
                
        pose = pose.to(DEVICE).float()
        all_gt_poses.append(pose)
        all_gt_mlengths.append(m_length)

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        em_pred_allgen_list.append(em_pred_allgen)
        
        # Store generated and real motions per class
        for j in range(bs):
            for jj in range(n_gen_samples):
                classwise_generated[label[j].item()].append(em_pred_allgen[jj*n_gen_samples + j].cpu().numpy())
            classwise_real[label[j].item()].append(em[j].cpu().numpy())

        temp_R_real = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match_real = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R_real
        matching_score_real += temp_match_real
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    em_pred_allgen_np = torch.cat(em_pred_allgen_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(em_pred_allgen_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    precision, recall = precision_and_recall(motion_pred_np, motion_annotation_np)
    
    all_gt_poses = torch.cat(all_gt_poses, dim=0).cpu().numpy()  # (num_gt_samples, max_seq_length, feature_dim)
    all_pred_poses = torch.cat(all_pred_poses, dim=0).cpu().numpy() # (num_pred_samples, max_seq_length, feature_dim)
    all_gt_mlengths = torch.cat(all_gt_mlengths, dim=0).cpu().numpy() # (num_gt_samples)
    all_pred_mlengths = torch.cat(all_pred_mlengths, dim=0).cpu().numpy() # (num_pred_samples)
    AVE_j, mean_AVE = cal_AVE(all_gt_poses, all_pred_poses, all_gt_mlengths, all_pred_mlengths)
    
    AVE = {'per_joint': AVE_j, 'mean': mean_AVE}
    
    print(f'number of generated samples: {em_pred_allgen_np.shape[0]}')

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}," \
          f"precision. {precision:.4f}, recall. {recall:.4f}," \
          f'AVE: {AVE["mean"]:.4f}'
    print(msg)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_label, real_motions in sorted(classwise_real.items()):
        if len(real_motions) > 0:
            real_np = np.array(real_motions)
            generated_np = np.array(classwise_generated[class_label])

            # class_diversity = calculate_diversity_allpairs_with_ci(generated_np)
            # class_diversity_real = calculate_diversity_allpairs_with_ci(real_np)
            d_mean, d_med, d_medci, l, u = calculate_diversity_allpairs_with_ci(generated_np)
            d_mean_r, d_med_r, d_medci_r, l_r, u_r = calculate_diversity_allpairs_with_ci(real_np)
            class_fid = calculate_frechet_distance(*calculate_activation_statistics(real_np),
                                                   *calculate_activation_statistics(generated_np))
            # class_R_precision = calculate_R_precision(real_np, generated_np, top_k=3, sum_all=True)
            # print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity: {class_diversity:.4f} (real: {class_diversity_real:.4f})")#, R_precision: {class_R_precision:.4f}")
            print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity mean: {d_mean:.4f} (real: {d_mean_r:.4f}), Diversity median: {d_med:.4f} (real: {d_med_r:.4f}), Diversity median (CI): {d_medci:.4f} (real: {d_medci_r:.4f}), Diversity CI: ({l:.4f}, {u:.4f}) (real: ({l_r:.4f}, {u_r:.4f}))")
            
            per_class_metrics[class_label] = {
                'fid': class_fid,
                'diversity_mean': d_mean,
                'diversity_real_mean': d_mean_r,
                'diversity_median': d_med,
                'diversity_real_median': d_med_r,
                'diversity_median_ci': d_medci,
                'diversity_real_median_ci': d_medci_r,
                'diversity_ci': (l, u),
                'diversity_real_ci': (l_r, u_r),
            }
            
    data_to_save = {
            'em_pred_allgen_np': em_pred_allgen_np,
            'motion_multimodality': motion_multimodality if not force_mask and cal_mm else None,
            'all_pred_poses': all_pred_poses,
            'all_pred_mlengths': all_pred_mlengths,
            'classwise_generated': classwise_generated
        }
    if out_path:
        with open(saved_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    return fid, diversity, diversity_real, R_precision, R_precision_real, matching_score_pred, matching_score_real, multimodality, precision, recall, per_class_metrics, AVE



@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_disent_prev(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)
                _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                  temperature=temperature, topk_filter_thres=topkr,
                                  force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))
            # pred_motions = vq_model.forward_decoder(mids)

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motions.clone(),
                                                              m_length)

        pose = pose.to(DEVICE).float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        
        # Store generated and real motions per class
        for j in range(bs):
            classwise_generated[label[j].item()].append(em_pred[j].cpu().numpy())
            classwise_real[label[j].item()].append(em[j].cpu().numpy())

        temp_R_real = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match_real = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R_real
        matching_score_real += temp_match_real
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    precision, recall = precision_and_recall(motion_pred_np, motion_annotation_np)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_label, real_motions in sorted(classwise_real.items()):
        if len(real_motions) > 0:
            real_np = np.array(real_motions)
            generated_np = np.array(classwise_generated[class_label])

            class_diversity = calculate_diversity(generated_np, min(300, len(generated_np)-1))
            class_diversity_real = calculate_diversity(real_np, min(300, len(real_np)-1))
            class_fid = calculate_frechet_distance(*calculate_activation_statistics(real_np),
                                                   *calculate_activation_statistics(generated_np))
            # class_R_precision = calculate_R_precision(real_np, generated_np, top_k=3, sum_all=True)
            print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity: {class_diversity:.4f} (real: {class_diversity_real:.4f})")#, R_precision: {class_R_precision:.4f}")
            
            per_class_metrics[class_label] = {
                'fid': class_fid,
                'diversity': class_diversity,
                'diversity_real': class_diversity_real,
                # 'R_precision': class_R_precision
            }

    return fid, diversity, diversity_real, R_precision, R_precision_real, matching_score_pred, matching_score_real, multimodality, per_class_metrics




@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_disent(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, out_path=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()
    generate = True
    saved_file_path = os.path.join(out_path, f'generated_data_{repeat_id}.pkl') if out_path else None
    if saved_file_path and os.path.exists(saved_file_path):
        generate = False

    if generate:
        print(f"Generating samples for evaluation. Repeat {repeat_id}")
        motion_annotation_list = []
        motion_pred_list = []
        em_pred_allgen_list =[]
        all_gt_poses = []
        all_pred_poses = []
        all_gt_mlengths = []
        all_pred_mlengths = []
        classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
        classwise_real = defaultdict(list)       # Dictionary to store real samples by class
        nb_sample = 0
        for i, batch in enumerate(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
            m_length = m_length.to(DEVICE)
            bs, seq = pose.shape[:2]
            # num_joints = 21 if pose.shape[-1] == 251 else 22
            em_pred_allgen = []
            n_gen_samples = 30
            for _ in range(n_gen_samples):
                _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                    temperature=temperature, topk_filter_thres=topkr, force_mask=force_mask)
                _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)
                em_pred_allgen.append(em_pred)
                all_pred_poses.append(pred_motions)
                all_pred_mlengths.append(m_length)
            em_pred_allgen = torch.cat(em_pred_allgen, dim=0)  # Shape: (bs * n_gen_samples, embedding_dim)
                    
            pose = pose.to(DEVICE).float()
            all_gt_poses.append(pose)
            all_gt_mlengths.append(m_length)

            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            motion_annotation_list.append(em)
            motion_pred_list.append(em_pred)
            em_pred_allgen_list.append(em_pred_allgen)
            
            # Store generated and real motions per class
            for j in range(bs):
                for jj in range(n_gen_samples):
                    classwise_generated[label[j].item()].append(em_pred_allgen[jj*n_gen_samples + j].cpu().numpy())
                classwise_real[label[j].item()].append(em[j].cpu().numpy())

            nb_sample += bs
        all_pred_poses = torch.cat(all_pred_poses, dim=0).cpu().numpy() # (num_pred_samples, max_seq_length, feature_dim)
        all_pred_mlengths = torch.cat(all_pred_mlengths, dim=0).cpu().numpy() # (num_pred_samples)
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        em_pred_allgen_np = torch.cat(em_pred_allgen_list, dim=0).cpu().numpy()
    else:
        print(f"Loading previously saved data from {saved_file_path}")
        with open(saved_file_path, 'rb') as f:
            saved_data = pickle.load(f)
        em_pred_allgen_np = saved_data['em_pred_allgen_np']
        all_pred_poses = saved_data['all_pred_poses']
        all_pred_mlengths = saved_data['all_pred_mlengths']
        classwise_generated = saved_data['classwise_generated']
        motion_pred_np = saved_data['motion_pred_np']
        motion_annotation_list = []
        all_gt_poses = []
        all_gt_mlengths = []
        classwise_real = defaultdict(list)       # Dictionary to store real samples by class
        nb_sample = 0
        for i, batch in enumerate(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
            m_length = m_length.to(DEVICE)
            bs, seq = pose.shape[:2]
            # num_joints = 21 if pose.shape[-1] == 251 else 22
            pose = pose.to(DEVICE).float()
            all_gt_poses.append(pose)
            all_gt_mlengths.append(m_length)
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            motion_annotation_list.append(em)
            # Store generated and real motions per class
            for j in range(bs):
                classwise_real[label[j].item()].append(em[j].cpu().numpy())
            nb_sample += bs
        
    all_gt_poses = torch.cat(all_gt_poses, dim=0).cpu().numpy()  # (num_gt_samples, max_seq_length, feature_dim)
    all_gt_mlengths = torch.cat(all_gt_mlengths, dim=0).cpu().numpy() # (num_gt_samples)
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(em_pred_allgen_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    precision, recall = precision_and_recall(motion_pred_np, motion_annotation_np)
    
    AVE_j, mean_AVE = cal_AVE(all_gt_poses, all_pred_poses, all_gt_mlengths, all_pred_mlengths)
    AVE = {'per_joint': AVE_j, 'mean': mean_AVE}
    
    print(f'number of generated samples: {em_pred_allgen_np.shape[0]}')
    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"precision. {precision:.4f}, recall. {recall:.4f}," \
          f'AVE: {AVE["mean"]:.4f}'
    print(msg)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_label, real_motions in sorted(classwise_real.items()):
        if len(real_motions) > 0:
            real_np = np.array(real_motions)
            generated_np = np.array(classwise_generated[class_label])

            # class_diversity = calculate_diversity_allpairs_with_ci(generated_np)
            # class_diversity_real = calculate_diversity_allpairs_with_ci(real_np)
            d_mean, d_med, d_medci, l, u = calculate_diversity_allpairs_with_ci(generated_np)
            d_mean_r, d_med_r, d_medci_r, l_r, u_r = calculate_diversity_allpairs_with_ci(real_np)
            class_fid = calculate_frechet_distance(*calculate_activation_statistics(real_np),
                                                   *calculate_activation_statistics(generated_np))
            # class_R_precision = calculate_R_precision(real_np, generated_np, top_k=3, sum_all=True)
            # print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity: {class_diversity:.4f} (real: {class_diversity_real:.4f})")#, R_precision: {class_R_precision:.4f}")
            print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity mean: {d_mean:.4f} (real: {d_mean_r:.4f}), Diversity median: {d_med:.4f} (real: {d_med_r:.4f}), Diversity median (CI): {d_medci:.4f} (real: {d_medci_r:.4f}), Diversity CI: ({l:.4f}, {u:.4f}) (real: ({l_r:.4f}, {u_r:.4f}))")
            
            per_class_metrics[class_label] = {
                'fid': class_fid,
                'diversity_mean': d_mean,
                'diversity_real_mean': d_mean_r,
                'diversity_median': d_med,
                'diversity_real_median': d_med_r,
                'diversity_median_ci': d_medci,
                'diversity_real_median_ci': d_medci_r,
                'diversity_ci': (l, u),
                'diversity_real_ci': (l_r, u_r),
            }
            
    data_to_save = {
            'em_pred_allgen_np': em_pred_allgen_np,
            'all_pred_poses': all_pred_poses,
            'all_pred_mlengths': all_pred_mlengths,
            'classwise_generated': classwise_generated,
            'motion_pred_np': motion_pred_np,
        }
    if out_path:
        with open(saved_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    return fid, diversity, diversity_real, precision, recall, per_class_metrics, AVE


@torch.no_grad()
def evaluation_geatfeatures_mask_transformer_test_plus_res_disent(train_loader, val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, out_path=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()
    
    all_gt_poses = []
    all_pred_poses = []
    all_gt_mlengths = []
    all_pred_mlengths = []
    all_label_gt = []
    all_label_pred = []
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class
    classwise_generated_len = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real_len = defaultdict(list)       # Dictionary to store real samples by class
    print('Generating features for *train and synthetic data*')
    for i, batch in enumerate(train_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)
        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        pred_poses = []
        pred_mlengths = []
        label_pred = []
        n_gen_samples = 1
        for _ in range(n_gen_samples):
            _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                temperature=temperature, topk_filter_thres=topkr,
                                force_mask=force_mask)
            _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))

            pred_poses.append(pred_motions)
            pred_mlengths.append(m_length)
            label_pred.append(label)
 
        pred_poses = torch.cat(pred_poses, dim=0)
        pred_mlengths = torch.cat(pred_mlengths, dim=0)
        label_pred = torch.cat(label_pred, dim=0)
        
        pose = pose.to(DEVICE).float()
        all_gt_poses.append(pose)
        all_gt_mlengths.append(m_length)
        all_label_gt.append(label)
        all_pred_poses.append(pred_poses)
        all_pred_mlengths.append(pred_mlengths)
        all_label_pred.append(label_pred)
        
        # Store generated and real motions per class
        for j in range(bs):
            for jj in range(n_gen_samples):
                classwise_generated[label[j].item()].append(pred_poses[jj*n_gen_samples + j].cpu().numpy())
                classwise_generated_len[label[j].item()].append(pred_mlengths[jj*n_gen_samples + j].cpu().numpy())
            classwise_real[label[j].item()].append(pose[j].cpu().numpy())
            classwise_real_len[label[j].item()].append(m_length[j].cpu().numpy())
            
    all_gt_poses = torch.cat(all_gt_poses, dim=0).cpu().numpy()  # (num_gt_samples, max_seq_length, feature_dim)
    all_pred_poses = torch.cat(all_pred_poses, dim=0).cpu().numpy() # (num_pred_samples, max_seq_length, feature_dim)
    all_gt_mlengths = torch.cat(all_gt_mlengths, dim=0).cpu().numpy() # (num_gt_samples)
    all_pred_mlengths = torch.cat(all_pred_mlengths, dim=0).cpu().numpy() # (num_pred_samples)
    all_label_gt = torch.cat(all_label_gt, dim=0).cpu().numpy() # (num_gt_samples)
    all_label_pred = torch.cat(all_label_pred, dim=0).cpu().numpy() # (num_pred_samples)    
    print(f'number of generated samples: {all_label_pred.shape[0]}')
            

    classwise_real_val = defaultdict(list)
    classwise_real_len_val = defaultdict(list)   
    print('Generating features for *validation data*')
    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)
        bs, seq = pose.shape[:2]
        pose = pose.to(DEVICE).float()
        # Store generated and real motions per class
        for j in range(bs):
            classwise_real_val[label[j].item()].append(pose[j].cpu().numpy())
            classwise_real_len_val[label[j].item()].append(m_length[j].cpu().numpy())
    
    extract_classwise_features(classwise_real, classwise_real_val, classwise_generated, classwise_real_len, classwise_real_len_val, classwise_generated_len, val_loader.dataset.inv_transform, out_path)





@torch.no_grad()
def evaluation_mask_transformer_test_plus_res_NOTdisent(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, out_path=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()
    generate = True
    saved_file_path = os.path.join(out_path, f'generated_data_{repeat_id}.pkl') if out_path else None
    if saved_file_path and os.path.exists(saved_file_path):
        generate = False

    if generate:
        print(f"Generating samples for evaluation. Repeat {repeat_id}")
        motion_annotation_list = []
        motion_pred_list = []
        em_pred_allgen_list =[]
        all_gt_poses = []
        all_pred_poses = []
        all_gt_mlengths = []
        all_pred_mlengths = []
        classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
        classwise_real = defaultdict(list)       # Dictionary to store real samples by class
        nb_sample = 0
        for i, batch in enumerate(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
            m_length = m_length.to(DEVICE)
            bs, seq = pose.shape[:2]
            # num_joints = 21 if pose.shape[-1] == 251 else 22
            em_pred_allgen = []
            n_gen_samples = 30
            for _ in range(n_gen_samples):
                mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)
                pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                pred_motions = vq_model.forward_decoder(pred_ids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(), m_length)
                em_pred_allgen.append(em_pred)
                all_pred_poses.append(pred_motions)
                all_pred_mlengths.append(m_length)
            em_pred_allgen = torch.cat(em_pred_allgen, dim=0)  # Shape: (bs * n_gen_samples, embedding_dim)
                    
            pose = pose.to(DEVICE).float()
            all_gt_poses.append(pose)
            all_gt_mlengths.append(m_length)

            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            motion_annotation_list.append(em)
            motion_pred_list.append(em_pred)
            em_pred_allgen_list.append(em_pred_allgen)
            
            # Store generated and real motions per class
            for j in range(bs):
                for jj in range(n_gen_samples):
                    classwise_generated[label[j].item()].append(em_pred_allgen[jj*n_gen_samples + j].cpu().numpy())
                classwise_real[label[j].item()].append(em[j].cpu().numpy())

            nb_sample += bs
        all_pred_poses = torch.cat(all_pred_poses, dim=0).cpu().numpy() # (num_pred_samples, max_seq_length, feature_dim)
        all_pred_mlengths = torch.cat(all_pred_mlengths, dim=0).cpu().numpy() # (num_pred_samples)
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        em_pred_allgen_np = torch.cat(em_pred_allgen_list, dim=0).cpu().numpy()
    else:
        print(f"Loading previously saved data from {saved_file_path}")
        with open(saved_file_path, 'rb') as f:
            saved_data = pickle.load(f)
        em_pred_allgen_np = saved_data['em_pred_allgen_np']
        all_pred_poses = saved_data['all_pred_poses']
        all_pred_mlengths = saved_data['all_pred_mlengths']
        classwise_generated = saved_data['classwise_generated']
        motion_pred_np = saved_data['motion_pred_np']
        motion_annotation_list = []
        all_gt_poses = []
        all_gt_mlengths = []
        classwise_real = defaultdict(list)       # Dictionary to store real samples by class
        nb_sample = 0
        for i, batch in enumerate(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
            m_length = m_length.to(DEVICE)
            bs, seq = pose.shape[:2]
            # num_joints = 21 if pose.shape[-1] == 251 else 22
            pose = pose.to(DEVICE).float()
            all_gt_poses.append(pose)
            all_gt_mlengths.append(m_length)
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
            motion_annotation_list.append(em)
            # Store generated and real motions per class
            for j in range(bs):
                classwise_real[label[j].item()].append(em[j].cpu().numpy())
            nb_sample += bs
        
    all_gt_poses = torch.cat(all_gt_poses, dim=0).cpu().numpy()  # (num_gt_samples, max_seq_length, feature_dim)
    all_gt_mlengths = torch.cat(all_gt_mlengths, dim=0).cpu().numpy() # (num_gt_samples)
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(em_pred_allgen_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    precision, recall = precision_and_recall(motion_pred_np, motion_annotation_np)
    
    AVE_j, mean_AVE = cal_AVE(all_gt_poses, all_pred_poses, all_gt_mlengths, all_pred_mlengths)
    AVE = {'per_joint': AVE_j, 'mean': mean_AVE}
    
    print(f'number of generated samples: {em_pred_allgen_np.shape[0]}')
    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"precision. {precision:.4f}, recall. {recall:.4f}," \
          f'AVE: {AVE["mean"]:.4f}'
    print(msg)
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for class_label, real_motions in sorted(classwise_real.items()):
        if len(real_motions) > 0:
            real_np = np.array(real_motions)
            generated_np = np.array(classwise_generated[class_label])

            # class_diversity = calculate_diversity_allpairs_with_ci(generated_np)
            # class_diversity_real = calculate_diversity_allpairs_with_ci(real_np)
            d_mean, d_med, d_medci, l, u = calculate_diversity_allpairs_with_ci(generated_np)
            d_mean_r, d_med_r, d_medci_r, l_r, u_r = calculate_diversity_allpairs_with_ci(real_np)
            class_fid = calculate_frechet_distance(*calculate_activation_statistics(real_np),
                                                   *calculate_activation_statistics(generated_np))
            # class_R_precision = calculate_R_precision(real_np, generated_np, top_k=3, sum_all=True)
            # print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity: {class_diversity:.4f} (real: {class_diversity_real:.4f})")#, R_precision: {class_R_precision:.4f}")
            print(f"-------> \tClass {class_label} Metrics -> FID: {class_fid:.4f}, Diversity mean: {d_mean:.4f} (real: {d_mean_r:.4f}), Diversity median: {d_med:.4f} (real: {d_med_r:.4f}), Diversity median (CI): {d_medci:.4f} (real: {d_medci_r:.4f}), Diversity CI: ({l:.4f}, {u:.4f}) (real: ({l_r:.4f}, {u_r:.4f}))")
            
            per_class_metrics[class_label] = {
                'fid': class_fid,
                'diversity_mean': d_mean,
                'diversity_real_mean': d_mean_r,
                'diversity_median': d_med,
                'diversity_real_median': d_med_r,
                'diversity_median_ci': d_medci,
                'diversity_real_median_ci': d_medci_r,
                'diversity_ci': (l, u),
                'diversity_real_ci': (l_r, u_r),
            }
            
    data_to_save = {
            'em_pred_allgen_np': em_pred_allgen_np,
            'all_pred_poses': all_pred_poses,
            'all_pred_mlengths': all_pred_mlengths,
            'classwise_generated': classwise_generated,
            'motion_pred_np': motion_pred_np,
        }
    if out_path:
        with open(saved_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

    return fid, diversity, diversity_real, precision, recall, per_class_metrics, AVE


@torch.no_grad()
def evaluation_geatfeatures_mask_transformer_test_plus_res_NOTdisent(train_loader, val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, out_path=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()
    
    all_gt_poses = []
    all_pred_poses = []
    all_gt_mlengths = []
    all_pred_mlengths = []
    all_label_gt = []
    all_label_pred = []
    
    classwise_generated = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real = defaultdict(list)       # Dictionary to store real samples by class
    classwise_generated_len = defaultdict(list)  # Dictionary to store generated samples by class
    classwise_real_len = defaultdict(list)       # Dictionary to store real samples by class
    print('Generating features for *train and synthetic data*')
    for i, batch in enumerate(train_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)
        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        pred_poses = []
        pred_mlengths = []
        label_pred = []
        n_gen_samples = 1
        for _ in range(n_gen_samples):
            mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                                      temperature=temperature, topk_filter_thres=topkr,
                                      gsample=gsample, force_mask=force_mask)
            pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            pred_motions = vq_model.forward_decoder(pred_ids)

            pred_poses.append(pred_motions)
            pred_mlengths.append(m_length)
            label_pred.append(label)
 
        pred_poses = torch.cat(pred_poses, dim=0)
        pred_mlengths = torch.cat(pred_mlengths, dim=0)
        label_pred = torch.cat(label_pred, dim=0)
        
        pose = pose.to(DEVICE).float()
        all_gt_poses.append(pose)
        all_gt_mlengths.append(m_length)
        all_label_gt.append(label)
        all_pred_poses.append(pred_poses)
        all_pred_mlengths.append(pred_mlengths)
        all_label_pred.append(label_pred)
        
        # Store generated and real motions per class
        for j in range(bs):
            for jj in range(n_gen_samples):
                classwise_generated[label[j].item()].append(pred_poses[jj*n_gen_samples + j].cpu().numpy())
                classwise_generated_len[label[j].item()].append(pred_mlengths[jj*n_gen_samples + j].cpu().numpy())
            classwise_real[label[j].item()].append(pose[j].cpu().numpy())
            classwise_real_len[label[j].item()].append(m_length[j].cpu().numpy())
            
    all_gt_poses = torch.cat(all_gt_poses, dim=0).cpu().numpy()  # (num_gt_samples, max_seq_length, feature_dim)
    all_pred_poses = torch.cat(all_pred_poses, dim=0).cpu().numpy() # (num_pred_samples, max_seq_length, feature_dim)
    all_gt_mlengths = torch.cat(all_gt_mlengths, dim=0).cpu().numpy() # (num_gt_samples)
    all_pred_mlengths = torch.cat(all_pred_mlengths, dim=0).cpu().numpy() # (num_pred_samples)
    all_label_gt = torch.cat(all_label_gt, dim=0).cpu().numpy() # (num_gt_samples)
    all_label_pred = torch.cat(all_label_pred, dim=0).cpu().numpy() # (num_pred_samples)    
    print(f'number of generated samples: {all_label_pred.shape[0]}')
            

    classwise_real_val = defaultdict(list)
    classwise_real_len_val = defaultdict(list)   
    print('Generating features for *validation data*')
    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, label = batch
        m_length = m_length.to(DEVICE)
        bs, seq = pose.shape[:2]
        pose = pose.to(DEVICE).float()
        # Store generated and real motions per class
        for j in range(bs):
            classwise_real_val[label[j].item()].append(pose[j].cpu().numpy())
            classwise_real_len_val[label[j].item()].append(m_length[j].cpu().numpy())
    
    extract_classwise_features(classwise_real, classwise_real_val, classwise_generated, classwise_real_len, classwise_real_len_val, classwise_generated_len, val_loader.dataset.inv_transform, out_path)





@torch.no_grad()
def Gen_4_downstream(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5, out_path=None):
    trans.eval()
    vq_model.eval()
    res_model.eval()
    saved_file_path = os.path.join(out_path, f'generated_data_downstream.pkl') if out_path else None

    all_pred_poses = []
    all_pred_mlengths = []
    all_pred_labels = []
    for i, batch in enumerate(val_loader):
        print(f'Batch {i}/{len(val_loader)}')
        _, _, clip_text, _, _, m_length, _, label = batch
        m_length = m_length.to(DEVICE)
        n_gen_samples = 2
        for _ in range(n_gen_samples):
            _, ids_m, ids_d = trans.generate(clip_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                temperature=temperature, topk_filter_thres=topkr, force_mask=force_mask)
            _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=label.to(DEVICE))
            all_pred_poses.append(pred_motions)
            all_pred_mlengths.append(m_length)
            all_pred_labels.append(label)
            
    extra_class_3_samples = 1  # Number of additional samples to generate per batch
    for i, batch in enumerate(val_loader):
        print(f'Generating additional class 3 samples for Batch {i}/{len(val_loader)}')
        _, _, clip_text, _, _, m_length, _, labels = batch  # Original batch data
        m_length = m_length.to(DEVICE)
        # Replace labels and clip_text for class 3
        modified_labels = torch.full_like(labels, 3)  # Replace all labels with 3
        modified_text = ["three"] * len(clip_text)  # Replace all clip_text with "three"
        for _ in range(extra_class_3_samples):
            # Generate data using modified labels and text for class 3
            _, ids_m, ids_d = trans.generate(modified_text, m_length // 4, m_length // 4, time_steps, cond_scale,
                                            temperature=temperature, topk_filter_thres=topkr, force_mask=force_mask)
            _, pred_ids_m, pred_ids_d = res_model.generate(ids_m, ids_d, modified_text, m_length // 4, temperature=1, 
                                                        cond_scale=res_cond_scale)
            pred_motions = vq_model.forward_decoder(pred_ids_m, pred_ids_d, y=modified_labels.to(DEVICE))

            # Append the additional samples for class 3
            all_pred_poses.append(pred_motions)
            all_pred_mlengths.append(m_length)  # Use the same m_length for consistency
            all_pred_labels.append(modified_labels)
        

    all_pred_poses = torch.cat(all_pred_poses, dim=0).cpu().numpy() # (num_pred_samples, max_seq_length, feature_dim)
    all_pred_mlengths = torch.cat(all_pred_mlengths, dim=0).cpu().numpy() # (num_pred_samples)
    all_pred_labels = torch.cat(all_pred_labels, dim=0).cpu().numpy() # (num_pred_samples)
    
    print(f'number of generated samples: {all_pred_labels.shape[0]}')
    unique_labels, counts = np.unique(all_pred_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples")

    data_to_save = {
            'all_pred_poses': all_pred_poses,
            'all_pred_mlengths': all_pred_mlengths,
            'all_pred_labels': all_pred_labels,
        }
    if out_path:
        with open(saved_file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    return all_pred_poses, all_pred_mlengths, all_pred_labels
