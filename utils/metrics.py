import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg
import torch
import json
from tqdm import tqdm
import torch.nn as nn
import math
from math import pi
from scipy.spatial import distance_matrix
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain as kinematic_chain
from scipy.ndimage import uniform_filter1d
import pickle
import seaborn as sns
import pandas as pd

def precision_and_recall(generated_features, real_features):
    k = 3

    data_num = min(len(generated_features), len(real_features))
    print(f'data num: {data_num}')

    if data_num <= 0:
        print("there is no data")
        return
    generated_features = generated_features[:data_num]
    real_features = real_features[:data_num]

    # get precision and recall
    precision = manifold_estimate(real_features, generated_features, k)
    recall = manifold_estimate(generated_features, real_features, k)

    return precision, recall

def manifold_estimate(A_features, B_features, k):
    A_features = np.array(A_features)  # Ensure input is a numpy array
    B_features = np.array(B_features)  # Ensure input is a numpy array
    KNN_list_in_A = {}

    for A in A_features: #tqdm(A_features, ncols=80):
        # Calculate pairwise distances with numpy norms
        pairwise_distances = np.linalg.norm(A_features - A, axis=1)
        
        # Find the k-th nearest distance
        v = np.partition(pairwise_distances, k)[k]
        KNN_list_in_A[tuple(A)] = v  # Using tuple(A) to use array as dictionary key

    n = 0

    for B in B_features:
        for A_prime in A_features:
            d = np.linalg.norm(B - A_prime)
            if d <= KNN_list_in_A[tuple(A_prime)]:
                n += 1
                break

    return n / len(B_features)



def calc_accel(preds, target):
    """
    Mean joint acceleration error
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    assert preds.dim() == 3
    # Expects BxJx3
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
    normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)
    accel_seq = normed.mean(1)
    return accel_seq


def calc_pampjpe(preds, target, sample_wise=True, return_transform_mat=False):
    # Expects BxJx3
    target, preds = target.float(), preds.float()
    # extracting the keypoints that all samples have valid annotations
    # valid_mask = (target[:, :, 0] != -2.).sum(0) == len(target)
    # preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(preds[:, valid_mask], target[:, valid_mask])
    # pa_mpjpe_each = compute_mpjpe(preds_tranformed, target[:, valid_mask], sample_wise=sample_wise)

    preds_tranformed, PA_transform = batch_compute_similarity_transform_torch(
        preds, target)
    pa_mpjpe_each = compute_mpjpe(preds_tranformed,
                                  target,
                                  sample_wise=sample_wise)

    if return_transform_mat:
        return pa_mpjpe_each, PA_transform
    else:
        return pa_mpjpe_each
    
def compute_mpjpe(preds,
                  target,
                  valid_mask=None,
                  pck_joints=None,
                  sample_wise=True):
    """
    Mean per-joint position error (i.e. mean Euclidean distance)
    often referred to as "Protocol #1" in many papers.
    """
    assert preds.shape == target.shape, print(preds.shape,
                                              target.shape)  # BxJx3
    mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

    if pck_joints is None:
        if sample_wise:
            mpjpe_seq = ((mpjpe * valid_mask.float()).sum(-1) /
                         valid_mask.float().sum(-1)
                         if valid_mask is not None else mpjpe.mean(-1))
        else:
            mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
        return mpjpe_seq
    else:
        mpjpe_pck_seq = mpjpe[:, pck_joints]
        return mpjpe_pck_seq


def align_by_parts(joints, align_inds=None):
    if align_inds is None:
        return joints
    pelvis = joints[:, align_inds].mean(1)
    return joints - torch.unsqueeze(pelvis, dim=1)


def calc_mpjpe(preds, target, align_inds=[0], sample_wise=True, trans=None):
    # Expects BxJx3
    valid_mask = target[:, :, 0] != -2.0
    # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
    if align_inds is not None:
        preds_aligned = align_by_parts(preds, align_inds=align_inds)
        if trans is not None:
            preds_aligned += trans
        target_aligned = align_by_parts(target, align_inds=align_inds)
    else:
        preds_aligned, target_aligned = preds, target
    mpjpe_each = compute_mpjpe(preds_aligned,
                               target_aligned,
                               valid_mask=valid_mask,
                               sample_wise=sample_wise)
    return mpjpe_each  # B

def batch_compute_similarity_transform_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat, (scale, R, t)
# --------------------------------------------------------------------------------

def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_diversity_allpairs_with_ci(activation, f=300):
    num_bootstrap_samples=1000
    ci_percentile=95
    assert len(activation.shape) == 2
    assert activation.shape[0] > 2
    num_samples = activation.shape[0]
    f = min(f, activation.shape[0])
    
    selected_indices = np.random.choice(num_samples, f, replace=False)
    sampled_activation = activation[selected_indices]
    
    pairwise_distances = distance_matrix(sampled_activation, sampled_activation)
    # Take only the upper triangular part, excluding the diagonal, to avoid double-counting and self-comparisons
    upper_triangle_distances = pairwise_distances[np.triu_indices(pairwise_distances.shape[0], k=1)]
    
    bootstrap_medians = []
    for _ in range(num_bootstrap_samples):
        resampled_distances = np.random.choice(upper_triangle_distances, len(upper_triangle_distances), replace=True)
        bootstrap_medians.append(np.median(resampled_distances))
        
    median_diversity = np.median(bootstrap_medians)
    lower_bound = np.percentile(bootstrap_medians, (100 - ci_percentile) / 2)
    upper_bound = np.percentile(bootstrap_medians, 100 - (100 - ci_percentile) / 2)

    return upper_triangle_distances.mean(),  np.median(upper_triangle_distances), median_diversity, lower_bound, upper_bound


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    
def cal_AVE(all_gt_poses, all_pred_poses, all_gt_mlengths, all_pred_mlengths, n_joints=22):
    """
    Computes the Average Variance Error (AVE) for local poses in the body's local coordinate system,
    without matching generated samples to ground truth samples.

    Parameters:
    - all_gt_poses: numpy array of shape (num_gt_samples, max_seq_length, feature_dim)
    - all_pred_poses: numpy array of shape (num_pred_samples, max_seq_length, feature_dim)
    - all_gt_mlengths: numpy array of shape (num_gt_samples,), sequence lengths for ground truth
    - all_pred_mlengths: numpy array of shape (num_pred_samples,), sequence lengths for generated data
    - n_joints: Total number of joints in the skeleton (including the root joint)

    Returns:
    - ave_error_per_joint: numpy array of shape (n_joints - 1,), per-joint AVE values
    - ave_error_mean: The mean AVE value across all joints (scalar)
    """
    
    start_idx = 4
    end_idx = start_idx + (n_joints - 1) * 3
    
    gt_local_poses = all_gt_poses[:, :, start_idx:end_idx]  # Shape: (num_gt_samples, max_seq_length, (n_joints - 1) * 3)
    gt_local_poses = gt_local_poses.reshape(-1, gt_local_poses.shape[1], n_joints - 1, 3)
    
    pred_local_poses = all_pred_poses[:, :, start_idx:end_idx]  # Shape: (num_pred_samples, max_seq_length, (n_joints - 1) * 3)
    pred_local_poses = pred_local_poses.reshape(-1, pred_local_poses.shape[1], n_joints - 1, 3)
    
    gt_variances_list = []
    pred_variances_list = []

    # Compute variances for ground truth data
    for i in range(len(all_gt_poses)):
        seq_length = int(all_gt_mlengths[i])
        gt_seq = gt_local_poses[i, :seq_length, :, :]  # Shape: (seq_length, n_joints - 1, 3)
        gt_var = np.var(gt_seq, axis=0, ddof=1)  # Shape: (n_joints - 1, 3)
        gt_variances_list.append(gt_var)

    # Compute variances for generated data
    for i in range(len(all_pred_poses)):
        seq_length = int(all_pred_mlengths[i])
        pred_seq = pred_local_poses[i, :seq_length, :, :]  # Shape: (seq_length, n_joints - 1, 3)
        pred_var = np.var(pred_seq, axis=0, ddof=1)  # Shape: (n_joints - 1, 3)
        pred_variances_list.append(pred_var)

    gt_variances = np.mean(np.stack(gt_variances_list), axis=0)  # Shape: (n_joints - 1, 3)
    pred_variances = np.mean(np.stack(pred_variances_list), axis=0)  # Shape: (n_joints - 1, 3)
    # Compute per-joint variance differences (absolute differences)
    variance_diffs = np.abs(gt_variances - pred_variances)  # Shape: (n_joints - 1, 3)
    # Average over coordinates to get per-joint AVE
    ave_error_per_joint = np.mean(variance_diffs, axis=1)  # Shape: (n_joints - 1,)
    # Compute mean AVE over all joints
    ave_error_mean = np.mean(ave_error_per_joint)

    return ave_error_per_joint, ave_error_mean



def extract_classwise_features(classwise_real, classwise_real_val, classwise_generated, classwise_real_len, classwise_real_len_val, classwise_generated_len, inv_transform, save_path_base):
    """
    Extract features per class for ground truth (classwise_real) and predictions (classwise_generated).
    classwise_real and classwise_generated are dictionaries with class labels as keys and lists of samples as values.
    each sample is a numpy array of shape (seq_length, 263).
    """
    # test_features()

    gt_train_features_per_class = {}
    gen_features_per_class = {}
    gt_train_features_per_class_all = {}
    gen_features_per_class_all = {}
    gt_val_features_per_class = {}
    gt_val_features_per_class_all = {}
    
    for class_label in classwise_real.keys():
        real_train_samples = np.array(classwise_real[class_label])
        generated_samples = np.array(classwise_generated[class_label])
        gt_train_mlengths = np.array(classwise_real_len[class_label])
        pred_mlengths = np.array(classwise_generated_len[class_label])
        real_val_samples = np.array(classwise_real_val[class_label])
        gt_val_mlengths = np.array(classwise_real_len_val[class_label])
        
        # Compute features for ground truth data (train)
        real_train_samples = inv_transform(real_train_samples)
        min_footclearance_list, min_armswing_list, stoop_list, stoop_ang_list = [], [], [], []
        for i, joint_data in enumerate(real_train_samples):
            joint_data = joint_data[:gt_train_mlengths[i]] # Shape: (seq_length, 263)
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy() # Shape: (seq_length, 22, 3)
            foot_clearance = calc_foot_lifting_relative(joint)
            armswing = calc_arm_swing_relative(joint)
            stoop = calc_stoop_posture_relative(joint)
            stoop_angle = calculate_stooped_posture_angle(joint)
            min_footclearance_list.append(np.min(foot_clearance))
            min_armswing_list.append(np.min(armswing))
            stoop_list.append(stoop)
            stoop_ang_list.append(stoop_angle)
        gt_train_features_per_class[class_label] = {
            'foot_clearance': np.mean(min_footclearance_list),
            'arm_swing': np.mean(min_armswing_list),
            'stoop_posture': np.mean(stoop_list),
            'stoop_angle': np.mean(stoop_ang_list),
            'number_samples': i+1,
        }
        gt_train_features_per_class_all[class_label] = {
            'foot_clearance': min_footclearance_list,
            'arm_swing': min_armswing_list,
            'stoop_posture': stoop_list,
            'stoop_angle': stoop_ang_list,
        }
        
        # Compute features for ground truth data (val)
        real_val_samples = inv_transform(real_val_samples)
        min_footclearance_list, min_armswing_list, stoop_list, stoop_ang_list = [], [], [], []
        for i, joint_data in enumerate(real_val_samples):
            joint_data = joint_data[:gt_val_mlengths[i]] # Shape: (seq_length, 263)
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy() # Shape: (seq_length, 22, 3)
            foot_clearance = calc_foot_lifting_relative(joint)
            armswing = calc_arm_swing_relative(joint)
            stoop = calc_stoop_posture_relative(joint)
            stoop_angle = calculate_stooped_posture_angle(joint)
            min_footclearance_list.append(np.min(foot_clearance))
            min_armswing_list.append(np.min(armswing))
            stoop_list.append(stoop)
            stoop_ang_list.append(stoop_angle)
        gt_val_features_per_class[class_label] = {
            'foot_clearance': np.mean(min_footclearance_list),
            'arm_swing': np.mean(min_armswing_list),
            'stoop_posture': np.mean(stoop_list),
            'stoop_angle': np.mean(stoop_ang_list),
            'number_samples': i+1,
        }
        gt_val_features_per_class_all[class_label] = {
            'foot_clearance': min_footclearance_list,
            'arm_swing': min_armswing_list,
            'stoop_posture': stoop_list,
            'stoop_angle': stoop_ang_list,
        }
        
        # Compute features for generated data
        generated_samples = inv_transform(generated_samples)
        min_footclearance_list, min_armswing_list, stoop_list, stoop_ang_list = [], [], [], []
        for i, joint_data in enumerate(generated_samples):
            joint_data = joint_data[:pred_mlengths[i]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
            foot_clearance = calc_foot_lifting_relative(joint)
            armswing = calc_arm_swing_relative(joint)
            stoop = calc_stoop_posture_relative(joint)
            stoop_angle = calculate_stooped_posture_angle(joint)
            min_footclearance_list.append(np.min(foot_clearance))
            min_armswing_list.append(np.min(armswing))
            stoop_list.append(stoop)
            stoop_ang_list.append(stoop_angle)
        gen_features_per_class[class_label] = {
            'foot_clearance': np.mean(min_footclearance_list),
            'arm_swing': np.mean(min_armswing_list),
            'stoop_posture': np.mean(stoop_list),
            'stoop_angle': np.mean(stoop_ang_list),
            'number_samples': i+1,
        }
        gen_features_per_class_all[class_label] = {
            'foot_clearance': min_footclearance_list,
            'arm_swing': min_armswing_list,
            'stoop_posture': stoop_list,
            'stoop_angle': stoop_ang_list,
        }
        
        gt_train_features_per_class = dict(sorted(gt_train_features_per_class.items()))
        gt_val_features_per_class = dict(sorted(gt_val_features_per_class.items()))
        gen_features_per_class = dict(sorted(gen_features_per_class.items()))
        gt_train_features_per_class_all = dict(sorted(gt_train_features_per_class_all.items()))
        gt_val_features_per_class_all = dict(sorted(gt_val_features_per_class_all.items()))
        gen_features_per_class_all = dict(sorted(gen_features_per_class_all.items()))

    
    pkl_path = os.path.join(save_path_base, 'classwise_features.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({"gt_train_features_per_class": gt_train_features_per_class,
                    "gt_val_features_per_class": gt_val_features_per_class,
                    "gen_features_per_class": gen_features_per_class,
                    "gt_train_features_per_class_all": gt_train_features_per_class_all,
                    "gt_val_features_per_class_all": gt_val_features_per_class_all,
                    "gen_features_per_class_all": gen_features_per_class_all,
                }, f)  
    save_path = os.path.join(save_path_base, 'classwise_features_bar_train.png')
    plot_classwise_features(gt_train_features_per_class, gen_features_per_class, save_path)
    save_path = os.path.join(save_path_base, 'classwise_features_bar_val.png')
    plot_classwise_features(gt_val_features_per_class, gen_features_per_class, save_path)

    data = {
        "Class": list(gt_train_features_per_class.keys()),
        "Ground Truth Features": list(gt_train_features_per_class.values()),
    }
    filename = os.path.join(save_path_base, 'classwise_features.txt')
    with open(filename, "a") as f:
        f.write(f"{'Class':<20} {'Ground Truth TRAIN Features':<30}\n")
        f.write("=" * 80 + "\n")
        for cls, gt_feature in zip(data["Class"], data["Ground Truth Features"]):
            f.write(f"{cls:<20} {str(gt_feature):<30}\n")
            
    data = {
        "Class": list(gt_train_features_per_class.keys()),
        "Ground Truth Features": list(gt_val_features_per_class.values()),
    }
    filename = os.path.join(save_path_base, 'classwise_features.txt')
    with open(filename, "a") as f:
        f.write(f"{'Class':<20} {'Ground Truth VAL Features':<30}\n")
        f.write("=" * 80 + "\n")
        for cls, gt_feature in zip(data["Class"], data["Ground Truth Features"]):
            f.write(f"{cls:<20} {str(gt_feature):<30}\n")

    data = {
        "Class": list(gt_train_features_per_class.keys()),
        "Generated Features": list(gen_features_per_class.values())
    }
    with open(filename, "a") as f:
        f.write(f"{'Class':<20}{'Generated Features':<30}\n")
        f.write("=" * 80 + "\n")
        for cls, gen_feature in zip(data["Class"], data["Generated Features"]):
            f.write(f"{cls:<20} {str(gen_feature):<30}\n")
            
    for feature in ['foot_clearance', 'arm_swing', 'stoop_posture', 'stoop_angle']:
        save_path = os.path.join(save_path_base, f'{feature}_boxplot_train.png')
        plot_feature_boxplots(gt_train_features_per_class_all, gen_features_per_class_all, feature, save_path)
    for feature in ['foot_clearance', 'arm_swing', 'stoop_posture', 'stoop_angle']:
        save_path = os.path.join(save_path_base, f'{feature}_boxplot_val.png')
        plot_feature_boxplots(gt_val_features_per_class_all, gen_features_per_class_all, feature, save_path)

            
    pp = 1


def plot_feature_boxplots(gt_features_per_class_all, gen_features_per_class_all, feature_name, save_path):
    """
    Creates box plots for GT and generated data side by side for each feature, optionally grouped by experiment.
    Parameters:
    - gt_features_per_class_all: Dictionary with class labels as keys and lists of GT values per feature.
    - gen_features_per_class_all: Dictionary with class labels as keys and lists of generated values per feature.
    - feature_name: The name of the feature to plot.
    - save_path: Path to save the plot image.
    """
    data = []
    for class_label in gt_features_per_class_all.keys():
        gt_data = gt_features_per_class_all[class_label][feature_name]
        gen_data = gen_features_per_class_all[class_label][feature_name]
        # Add GT data
        for value in gt_data:
            data.append({'Value': value, 'Type': 'GT', 'Class': class_label})
        # Add Generated data
        for value in gen_data:
            data.append({'Value': value, 'Type': 'Generated', 'Class': class_label})
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Value', hue='Type', data=df, palette="Set2")
    plt.title(f"{feature_name} Box Plot by Class")
    plt.xlabel("Class")
    plt.ylabel(feature_name)
    plt.legend(title="Data Type")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

    
def plot_classwise_features(gt_features_per_class, gen_features_per_class, save_path):
    # Extract classes and features
    classes = sorted(gt_features_per_class.keys())
    features = ['foot_clearance', 'arm_swing', 'stoop_posture', 'stoop_angle']

    # Prepare data for plotting
    gt_means = {feature: [gt_features_per_class[c][feature] for c in classes] for feature in features}
    gen_means = {feature: [gen_features_per_class[c][feature] for c in classes] for feature in features}

    # Plot each feature as a grouped bar plot
    x = np.arange(len(classes))
    width = 0.35

    fig, axs = plt.subplots(len(features), 1, figsize=(10, 4 * len(features)))
    for i, feature in enumerate(features):
        ax = axs[i]
        ax.bar(x - width/2, gt_means[feature], width, label='GT')
        ax.bar(x + width/2, gen_means[feature], width, label='Generated')

        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_title(f'{feature.capitalize()} Comparison per Class')
        ax.set_xlabel('Class')
        ax.set_ylabel(f'{feature.capitalize()}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_radar(gt_features_per_class, gen_features_per_class, save_path):
    features = ['foot_clearance', 'arm_swing', 'stoop_posture', 'stoop_angle']
    num_vars = len(features)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Close the plot

    # Create subplots, one per class
    num_classes = len(gt_features_per_class)
    fig, axs = plt.subplots(1, num_classes, subplot_kw=dict(polar=True), figsize=(5 * num_classes, 5))
    fig.suptitle("Comparison of GT and Generated Features by Class")

    # Ensure axs is iterable (convert to list if there's only one subplot)
    if num_classes == 1:
        axs = [axs]

    for idx, (class_label, gt_values) in enumerate(gt_features_per_class.items()):
        gen_values = gen_features_per_class[class_label]

        # Prepare data for GT and generated features
        gt_data = [gt_values[feature] for feature in features] + [gt_values[features[0]]]
        gen_data = [gen_values[feature] for feature in features] + [gen_values[features[0]]]

        # Plot on radar
        ax = axs[idx]
        ax.plot(angles, gt_data, linewidth=2, linestyle='solid', label='GT')
        ax.fill(angles, gt_data, alpha=0.25)
        ax.plot(angles, gen_data, linewidth=2, linestyle='solid', label='Generated')
        ax.fill(angles, gen_data, alpha=0.25)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.set_title(f'Class {class_label}')
        ax.legend()

    plt.savefig(save_path)
    plt.close()

joint_map = { "lhip": 1, "rhip": 2,
            "lank": 7, "rank": 8,
            "sacrum": 0, "neck": 12,
            "lwrist": 20, "rwrist": 21,
            "lshoulder": 16, "rshoulder": 17}

def calc_stoop_posture(pose_seq, window_size=5):
    leg_len = leg_length(pose_seq)
        
    sacrum_positions = pose_seq[:, joint_map['sacrum'], :]
    neck_positions = pose_seq[:, joint_map['neck'], :]
    
    # Calculate forward direction for each frame and apply smoothing
    forward_direction = sacrum_positions[1:] - sacrum_positions[:-1]
    forward_direction = np.vstack([forward_direction[0]] * window_size + list(forward_direction))
    # Smooth each dimension of the forward_direction
    forward_direction_smooth = np.empty_like(forward_direction)
    for i in range(forward_direction.shape[1]):
        forward_direction_smooth[:, i] = uniform_filter1d(forward_direction[:, i], size=window_size)
    forward_direction = forward_direction_smooth[-(len(sacrum_positions) - 1):]
    forward_direction = forward_direction / np.linalg.norm(forward_direction, axis=1, keepdims=True)
    
    # Project neck to sacrum vector onto averaged forward direction
    neck_to_sacrum = neck_positions[:-1] - sacrum_positions[:-1]
    horizontal_distance = np.einsum('ij,ij->i', neck_to_sacrum[:len(forward_direction)], forward_direction) / leg_len
    
    return np.mean(horizontal_distance)

def calc_arm_swing(pose_seq, window_size=5):
    """Calculates the arm swing range relative to the forward direction,
    averaging from both the left and right arms.
    Returns:
    - List with [right_arm_swing, left_arm_swing] normalized by leg length.
    """
    leg_len = leg_length(pose_seq)

    sacrum_positions = pose_seq[:, joint_map['sacrum'], :]
    rwrist_positions = pose_seq[:, joint_map['rwrist'], :]
    lwrist_positions = pose_seq[:, joint_map['lwrist'], :]
    
    # Calculate forward direction with smoothing
    forward_direction = sacrum_positions[1:] - sacrum_positions[:-1]
    forward_direction = np.vstack([forward_direction[0]] * window_size + list(forward_direction))
    # Smooth each dimension of the forward_direction
    forward_direction_smooth = np.empty_like(forward_direction)
    for i in range(forward_direction.shape[1]):
        forward_direction_smooth[:, i] = uniform_filter1d(forward_direction[:, i], size=window_size)
    forward_direction = forward_direction_smooth[-(len(sacrum_positions) - 1):]
    forward_direction = forward_direction / np.linalg.norm(forward_direction, axis=1, keepdims=True)
    
    # Center wrists relative to sacrum position
    rwrist_to_sacrum = rwrist_positions[:-1] - sacrum_positions[:-1]
    lwrist_to_sacrum = lwrist_positions[:-1] - sacrum_positions[:-1]
    
    # Project wrist movements onto forward direction
    right_arm_swing = np.einsum('ij,ij->i', rwrist_to_sacrum[:len(forward_direction)], forward_direction) / leg_len
    left_arm_swing = np.einsum('ij,ij->i', lwrist_to_sacrum[:len(forward_direction)], forward_direction) / leg_len
    
    # Calculate swing range for each arm
    arm_swing_range_r = np.max(right_arm_swing) - np.min(right_arm_swing)
    arm_swing_range_l = np.max(left_arm_swing) - np.min(left_arm_swing)

    return [arm_swing_range_r, arm_swing_range_l]


def calc_foot_lifting(pose_seq):
    """ Assumes y dimension of coordinates corresponds to height
        and is at index 1 of each coordinate. 

        Calculates the range of the y coordinate for the ankle 
        scaled by the length of the femur bone
    """
    leg_len = leg_length(pose_seq)
    foot_height_r = pose_seq[:, joint_map["rank"],1]
    foot_height_l = pose_seq[:, joint_map["lank"],1]

    foot_lifting_range_r = (np.max(foot_height_r) - np.min(foot_height_r)) / leg_len
    foot_lifting_range_l = (np.max(foot_height_l) - np.min(foot_height_l)) / leg_len

    return [foot_lifting_range_r, foot_lifting_range_l]

def leg_length(pose_seq):
    """Calculates the mean leg length. Avg of both left and right leg length 
    (average of the maximum observed leg length across frames, which typically occurs during the heel strike phase when the leg is straight).
    Distance from hip loc to foot loc.
    """
    left_hip_locs = pose_seq[:, joint_map["lhip"], :]
    right_hip_locs = pose_seq[:, joint_map["rhip"], :]
    left_foot_locs = pose_seq[:, joint_map["lank"], :]
    right_foot_locs = pose_seq[:, joint_map["rank"], :]

    # Calculate the Euclidean norm (vector norm) for right and left legs
    right_leg_length = np.linalg.norm(right_hip_locs - right_foot_locs, axis=1)
    left_leg_length = np.linalg.norm(left_hip_locs - left_foot_locs, axis=1)
    
    r_max = np.max(right_leg_length)
    l_max = np.max(left_leg_length)

    return np.mean([r_max, l_max])



def calc_arm_swing_relative(pose_seq):
    """Calculates the arm swing range using relative movements between wrist and shoulder joints.
    Returns:
    - List with [right_arm_swing, left_arm_swing] normalized by leg length.
    """
    leg_len = leg_length(pose_seq)

    # Get wrist and shoulder positions
    rwrist_positions = pose_seq[:, joint_map['rwrist'], :]
    lwrist_positions = pose_seq[:, joint_map['lwrist'], :]
    rshoulder_positions = pose_seq[:, joint_map['rshoulder'], :]
    lshoulder_positions = pose_seq[:, joint_map['lshoulder'], :]

    # Calculate relative wrist-to-shoulder distances for both arms
    rwrist_to_shoulder = rwrist_positions - rshoulder_positions
    lwrist_to_shoulder = lwrist_positions - lshoulder_positions

    # Calculate the swing range (max - min) for each axis independently, then normalize by leg length
    arm_swing_range_r = (np.max(rwrist_to_shoulder, axis=0) - np.min(rwrist_to_shoulder, axis=0)) / leg_len
    arm_swing_range_l = (np.max(lwrist_to_shoulder, axis=0) - np.min(lwrist_to_shoulder, axis=0)) / leg_len

    # Return the total swing range (sum of x, y, z ranges) for both arms
    return [np.sum(arm_swing_range_r), np.sum(arm_swing_range_l)]


def calc_stoop_posture_relative(pose_seq):
    """Calculates the stoop posture using the vertical distance (y-axis) between the neck and sacrum.
    Returns:
    - A single value representing the normalized stoop posture.
    """
    leg_len = leg_length(pose_seq)

    # Get sacrum and neck positions
    sacrum_positions = pose_seq[:, joint_map['sacrum'], :]
    neck_positions = pose_seq[:, joint_map['neck'], :]

    # Calculate the vertical (y-axis) distance between neck and sacrum
    stoop_distance_y = neck_positions[:, 1] - sacrum_positions[:, 1]

    # Return the average stoop distance normalized by leg length
    return np.mean(stoop_distance_y) / leg_len


def calc_foot_lifting_relative(pose_seq):
    """Calculates the foot clearance based on the vertical range of the ankle joints.
    Assumes the y-axis represents height.
    Returns:
    - List with [right_foot_clearance, left_foot_clearance] normalized by leg length.
    """
    leg_len = leg_length(pose_seq)

    # Get ankle positions
    right_foot_height = pose_seq[:, joint_map["rank"], 1]
    left_foot_height = pose_seq[:, joint_map["lank"], 1]

    # Calculate the range of foot lifting for each foot
    foot_lifting_range_r = (np.max(right_foot_height) - np.min(right_foot_height)) / leg_len
    foot_lifting_range_l = (np.max(left_foot_height) - np.min(left_foot_height)) / leg_len

    return [foot_lifting_range_r, foot_lifting_range_l]

def calculate_stooped_posture_angle(pose_seq):
    """
    Calculate the stooped posture angle based on the neck, sacrum, and feet positions
    over a sequence of poses.

    Parameters:
    - pose_seq (np.array): Sequence of 3D coordinates for each joint, with shape 
      [num_frames, num_joints, 3].

    Returns:
    - float: The average angle in degrees indicating the stooped posture across frames.
             Smaller angles represent a more pronounced stoop.
    """
    sacrum_positions = pose_seq[:, joint_map['sacrum'], :]
    neck_positions = pose_seq[:, joint_map['neck'], :]
    lfeet_positions = pose_seq[:, joint_map['lank'], :]
    rfeet_positions = pose_seq[:, joint_map['rank'], :]
    feet = (lfeet_positions + rfeet_positions) / 2  # Midpoint between left and right feet
    
    # Calculate vectors for neck-sacrum and sacrum-feet
    vector_neck_to_sacrum = neck_positions - sacrum_positions
    vector_sacrum_to_feet = feet - sacrum_positions
    
    # Calculate dot products and magnitudes frame by frame
    dot_product = np.einsum('ij,ij->i', vector_neck_to_sacrum, vector_sacrum_to_feet)
    magnitude_neck_to_sacrum = np.linalg.norm(vector_neck_to_sacrum, axis=1)
    magnitude_sacrum_to_feet = np.linalg.norm(vector_sacrum_to_feet, axis=1)
    
    # Calculate the cosine of the angle and then the angle in degrees
    cos_angle = dot_product / (magnitude_neck_to_sacrum * magnitude_sacrum_to_feet)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Clip to handle numerical errors
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # Return the average angle over the sequence
    return np.min(angle_deg)
