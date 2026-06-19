import torch
import torch.nn as nn
import torch.nn.functional as F

class MDWALoss(nn.Module):
    def __init__(self, alpha, beta):
        super(MDWALoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, logits, labels):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        # print('probs:\n', probs)
        
        # Get the log probabilities
        log_probs = torch.log(probs + 1e-8)  # Adding epsilon to avoid log(0)
        # print('log_probs:\n', log_probs)

        N, C = logits.size() # N: batch size, C: number of classes
        one_hot_labels = F.one_hot(labels, num_classes=C).float()


        log_p_c = (log_probs * one_hot_labels).sum(dim=1, keepdim=True)
        p_c = (probs * one_hot_labels).sum(dim=1)

        # Prepare the mask for p_(c-1) and p_(c+1)
        mask_alpha = (p_c < self.alpha).float().unsqueeze(1).expand_as(log_probs)
        # Create shifted logits for p_(c-1) and p_(c+1)
        log_p_c_minus_1 = torch.cat([log_probs[:, 1:], torch.full((N, 1), float('-inf')).to(logits.device)], dim=1)
        log_p_c_plus_1 = torch.cat([torch.full((N, 1), float('-inf')).to(logits.device), log_probs[:, :-1]], dim=1)

        log_p_c_repeated = log_p_c.expand_as(log_p_c_minus_1)
        
        # Calculate max of neighbor probabilities
        p_c_minus_1 = torch.cat([probs[:, 1:], torch.zeros((N, 1)).to(logits.device)], dim=1)
        p_c_plus_1 = torch.cat([torch.zeros((N, 1)).to(logits.device), probs[:, :-1]], dim=1)
        max_p_neighbors = torch.max(p_c_minus_1, p_c_plus_1)
        mask_beta = (max_p_neighbors > self.beta).float()
        
        # print(torch.stack([log_p_c_minus_1, log_p_c_repeated, log_p_c_plus_1], dim=2))
        # Calculate p_ms
        p_m = mask_alpha * mask_beta * torch.max(torch.stack([log_p_c_minus_1, log_p_c_repeated, log_p_c_plus_1], dim=2), dim=2)[0]+ \
                (1 - mask_alpha) * log_p_c_repeated + \
                mask_alpha * (1 - mask_beta) * log_p_c_repeated
        # p_m = mask * torch.max(torch.stack([log_p_c_minus_1, log_p_c_repeated, log_p_c_plus_1], dim=2), dim=2)[0] + (1 - mask) * log_p_c_repeated

        p_m = torch.where(p_m == float('-inf'), torch.full_like(p_m, -10000.0), p_m)
        # print('one_hot_labels * p_m:\n', one_hot_labels * p_m)
        loss = -(one_hot_labels * p_m).sum() / N

        return loss

def rot6d_to_rotmat(x):
    """
    Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt process.
    Args:
        x: Tensor of shape (..., 6)
    Returns:
        R: Rotation matrix of shape (..., 3, 3)
    """
    x = x.view(-1, 6)
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    b1 = F.normalize(a1, dim=1)
    # print(f'b1: {b1}')
    dot_prod = torch.sum(b1 * a2, dim=1, keepdim=True)
    # print(f'dot_prod: {dot_prod}')
    a2_ortho = a2 - dot_prod * b1
    # print(f'a2_ortho: {a2_ortho}')
    b2 = F.normalize(a2_ortho, dim=1)
    # print(f'b2: {b2}')
    b3 = torch.cross(b1, b2, dim=1)
    # print(f'b3: {b3}')
    R = torch.stack([b1, b2, b3], dim=-1)  # Shape: (batch_size * num_joints, 3, 3)
    return R


def geodesic_loss(R_pred, R_target):
    # R_pred.shape: (B*T*(num_joints - 1), 3, 3)
    R_rel = torch.matmul(R_pred.transpose(-2, -1), R_target)
    cos_theta = (R_rel.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    loss = theta.mean()
    # print('geodesic_loss: cos_theta', cos_theta)
    # print('geodesic_loss: theta', theta)
    # print('geodesic_loss: loss', loss)
    return loss

def calc_geodesic_loss(pred_rot_6d, target_rot_6d, num_joints):
    # pred_rot_6d.shape: (B, T, (num_joints - 1)*6 = 126)
    batch_size, seq_len, _ = target_rot_6d.shape
    target_rot_6d_flat = target_rot_6d.reshape(-1, 6) # target_rot_6d_flat.shape: (B*T*(num_joints - 1), 6)
    pred_rot_6d_flat = pred_rot_6d.reshape(-1, 6)
    target_rot_matrices = rot6d_to_rotmat(target_rot_6d_flat) # target_rot_matrices.shape: (B*T*(num_joints - 1), 3, 3)
    pred_rot_matrices = rot6d_to_rotmat(pred_rot_6d_flat)
    rotation_loss = geodesic_loss(pred_rot_matrices, target_rot_matrices)
    # print('calc_geodesic_loss: rotation_loss', rotation_loss)
    return rotation_loss

def cal_geodesic_l1_loss(pred_motion, motions, num_joints, l1_criterion):
    rot_start_idx = 4 + (num_joints - 1) * 3 
    rot_end_idx = rot_start_idx + (num_joints - 1) * 6
    
    target_non_rot_part1 = motions[..., :rot_start_idx]
    target_non_rot_part2 = motions[..., rot_end_idx:]
    target_non_rot = torch.cat([target_non_rot_part1, target_non_rot_part2], dim=-1)
    pred_non_rot_part1 = pred_motion[..., :rot_start_idx]
    pred_non_rot_part2 = pred_motion[..., rot_end_idx:]
    pred_non_rot = torch.cat([pred_non_rot_part1, pred_non_rot_part2], dim=-1)
    
    loss_non_rot = l1_criterion(pred_non_rot, target_non_rot)
    # print('cal_geodesic_l1_loss: loss_non_rot', loss_non_rot)
    
    target_rot_6d = motions[..., rot_start_idx:rot_end_idx]
    pred_rot_6d = pred_motion[..., rot_start_idx:rot_end_idx]
    
    assert num_joints - 1 == (rot_end_idx - rot_start_idx) // 6, 'Number of joints mismatch'
    rotation_loss = calc_geodesic_loss(pred_rot_6d, target_rot_6d, num_joints - 1)
    # print('cal_geodesic_l1_loss: rotation_loss', rotation_loss)
    
    total_loss = loss_non_rot + rotation_loss
    # print('cal_geodesic_l1_loss: total_loss', total_loss)
    return total_loss
    
if __name__ == '__main__':
    # Test the loss function
    loss_fn = MDWALoss(alpha=0.2, beta=0.8)
    logits = torch.tensor([[2.0, 1.0, 0.1, 0.5], [0.5, 2.5, 0.4, 0.6], [1.5, 0.15, 1.5, 0.06]])
    labels = torch.tensor([0, 2, 1])
    loss = loss_fn(logits, labels)
    print('loss:', loss)