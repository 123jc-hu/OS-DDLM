import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def entropy_loss(logits):
    """计算交叉熵信息熵损失"""
    probs = F.softmax(logits, dim=-1)  # 计算 softmax 概率
    log_probs = F.log_softmax(logits, dim=-1)  # 计算对数概率
    entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))  # 计算熵
    return entropy

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, d=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        if d is not None:
            d = d.contiguous().view(-1, 1)
            mask_d = (~torch.eq(d, d.T)).float().to(device)
            mask_d = mask_d.repeat(anchor_count, contrast_count)
            mask = mask * logits_mask * mask_d
        else:
            mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        def finite_mean(x):
            # only 1D for now
            num_finite = (torch.isfinite(x).float()).sum()
            mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
            if num_finite != 0:
                mean = mean / num_finite
            else:
                return torch.tensor(0.0).to(x)
            return mean

        return loss
    
def center_cosine_distance_loss(features, labels):
    """
    计算每个样本到其类中心的余弦距离

    参数:
        features (Tensor): 特征矩阵，形状为 (B, d)
        labels (Tensor): 标签，形状为 (B,)

    返回:
        loss (Tensor): 余弦距离的均值
    """
    # 获取类别信息
    classes = torch.unique(labels)
    num_classes = len(classes)
    num_features = features.size(1)

    # 初始化类中心矩阵
    centers = torch.zeros(num_classes, num_features, device=features.device)

    # 计算每个类的中心（均值）
    for i, c in enumerate(classes):
        mask = (labels == c)
        class_features = features[mask]
        if len(class_features) > 0:
            centers[i] = class_features.mean(dim=0)

    # 为每个样本找到对应的类中心
    batch_centers = centers[labels]  # shape=(B, d)

    # 计算余弦相似度（范围 [-1, 1]）
    cosine_sim = F.cosine_similarity(features, batch_centers, dim=1)  # shape=(B,)

    # 余弦距离 = 1 - 余弦相似度（范围 [0, 2]）
    cosine_dist = 1 - cosine_sim

    # 计算平均余弦距离作为 loss
    loss = torch.mean(cosine_dist)

    return loss

class CrossSessionCenterAlignmentLoss(nn.Module):
    """
    用于跨session的EEG特征对齐：
    - 同类别特征收敛 (Center Loss)
    - 跨session同类别中心余弦相似度最大化 (Domain Alignment)
    假设特征已经归一化。
    """
    def __init__(self, lambda_center=1.0, lambda_align=1.0):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_align = lambda_align

    def forward(self, features, labels, sessions):
        """
        features: (B, D) 已归一化的特征向量
        labels:   (B,)   类别标签 (0/1)
        sessions: (B,)   session标签 (0/1/n)
        """
        centers = self._compute_centers(features, labels, sessions)

        center_loss = self._compute_center_loss(features, labels, sessions, centers)
        align_loss = self._compute_domain_align_loss(centers)

        total_loss = self.lambda_center * center_loss + self.lambda_align * align_loss
        return total_loss, center_loss, align_loss

    @staticmethod
    def _compute_centers(features, labels, sessions):
        """按类别和session计算特征中心"""
        centers = {}
        unique_labels = labels.unique()
        unique_sessions = sessions.unique()

        for y in unique_labels:
            for s in unique_sessions:
                mask = (labels == y) & (sessions == s)
                if mask.sum() > 0:
                    centers[(int(y.item()), int(s.item()))] = features[mask].mean(dim=0)
        return centers

    @staticmethod
    def _compute_center_loss(features, labels, sessions, centers):
        """计算每个样本与其中心的距离，特征已归一化，距离用(1 - 余弦相似度)"""
        loss = 0.0
        for i in range(features.size(0)):
            key = (int(labels[i].item()), int(sessions[i].item()))
            c = centers[key]
            cos_sim = F.cosine_similarity(features[i].unsqueeze(0), c.unsqueeze(0))
            loss += (1 - cos_sim)  # 越接近越好
        return loss / features.size(0)

    # @staticmethod
    # def _compute_domain_align_loss(centers):
    #     """跨session同类特征中心，余弦相似度最大化"""
    #     loss = 0.0
    #     class_session = {}

    #     # 先按类别收集所有session的中心
    #     for (label, session), center in centers.items():
    #         class_session.setdefault(label, []).append((session, center))

    #     # 计算每一类的session间两两相似度
    #     for label, session_centers in class_session.items():
    #         for i in range(len(session_centers)):
    #             for j in range(i + 1, len(session_centers)):
    #                 c1 = session_centers[i][1]
    #                 c2 = session_centers[j][1]
    #                 cos_sim = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0))
    #                 loss += (1 - cos_sim)  # 越相似越小

    #     if len(class_session) > 0:
    #         num_pairs = sum(len(v) * (len(v)-1) / 2 for v in class_session.values())
    #         loss = loss / (num_pairs + 1e-6)  # 避免除0
    #     return loss

    @staticmethod
    def _compute_domain_align_loss(centers):
        """
        不再 domain 两两对齐，而是把每类所有 session
        收缩到该类的 global prototype
        """
        loss = 0.0
        for label in {k[0] for k in centers.keys()}:
            # 收集该类所有 session center
            cls_centers = [c for (y, _), c in centers.items() if y == label]
            if len(cls_centers) < 2:
                continue
            proto = torch.stack(cls_centers).mean(dim=0)          # global
            for c in cls_centers:
                loss += 1 - F.cosine_similarity(
                    c.unsqueeze(0), proto.unsqueeze(0)
                )
            loss = loss / len(cls_centers)
        return loss


class CrossSessionCenterAlignMarginLoss(nn.Module):
    """
    跨Session EEG特征三合一损失:
    1. Center Loss (余弦距离)
    2. Align Loss (跨Session同类中心相似)
    3. Angular Margin Loss (类间中心角度最大化)

    特征应已归一化。
    """
    def __init__(self, lambda_center=1.0, lambda_align=0.1, lambda_margin=0.05):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_align = lambda_align
        self.lambda_margin = lambda_margin

    def forward(self, features, labels, sessions):
        """
        features: (B, D) 已归一化特征向量
        labels:   (B,)   类别标签 0/1
        sessions: (B,)   Session标签 0/1
        """
        centers = self._compute_centers(features, labels, sessions)

        center_loss = self._compute_center_loss(features, labels, sessions, centers)
        align_loss = self._compute_domain_align_loss(centers)
        margin_loss = self._compute_angular_margin_loss(centers)

        total_loss = (
            self.lambda_center * center_loss +
            self.lambda_align * align_loss +
            self.lambda_margin * margin_loss
        )

        return total_loss, center_loss, align_loss, margin_loss

    @staticmethod
    def _compute_centers(features, labels, sessions):
        centers = {}
        unique_labels = labels.unique()
        unique_sessions = sessions.unique()

        for y in unique_labels:
            for s in unique_sessions:
                mask = (labels == y) & (sessions == s)
                if mask.sum() > 0:
                    centers[(int(y.item()), int(s.item()))] = features[mask].mean(dim=0)
        return centers

    @staticmethod
    def _compute_center_loss(features, labels, sessions, centers):
        loss = 0.0
        for i in range(features.size(0)):
            key = (int(labels[i].item()), int(sessions[i].item()))
            c = centers[key]
            cos_sim = F.cosine_similarity(features[i].unsqueeze(0), c.unsqueeze(0))
            loss += (1 - cos_sim)
        return loss / features.size(0)

    @staticmethod
    def _compute_domain_align_loss(centers):
        loss = 0.0
        class_session = {}
        for (label, session), center in centers.items():
            class_session.setdefault(label, []).append((session, center))

        for label, session_centers in class_session.items():
            for i in range(len(session_centers)):
                for j in range(i + 1, len(session_centers)):
                    c1 = session_centers[i][1]
                    c2 = session_centers[j][1]
                    cos_sim = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0))
                    loss += (1 - cos_sim)

        num_pairs = sum(len(v) * (len(v) - 1) / 2 for v in class_session.values())
        if num_pairs > 0:
            loss = loss / num_pairs
        return loss

    @staticmethod
    def _compute_angular_margin_loss(centers):
        """
        类间中心的夹角，尽量让不同类中心相距更远。
        """
        loss = 0.0
        class_to_centers = {}
        for (label, session), center in centers.items():
            class_to_centers.setdefault(label, []).append(center)

        unique_labels = list(class_to_centers.keys())
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                for c1 in class_to_centers[unique_labels[i]]:
                    for c2 in class_to_centers[unique_labels[j]]:
                        cos_sim = F.cosine_similarity(c1.unsqueeze(0), c2.unsqueeze(0))
                        loss += cos_sim  # 越相似越罚

        num_pairs = 0
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                num_pairs += len(class_to_centers[unique_labels[i]]) * len(class_to_centers[unique_labels[j]])

        if num_pairs > 0:
            loss = loss / num_pairs
        return loss
    
    
class ContrastiveLoss_for_TSformer_SA(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = torch.tensor(temperature)
        
    def forward(self, emb_i, emb_j):	
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        batch_size = emb_i.shape[0]
        negatives_mask = 1 - torch.eye(batch_size * 2, batch_size * 2, dtype=bool).float()

        representations = torch.cat([z_i, z_j], dim=0)          #(2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = (negatives_mask).to(emb_i.device) * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))     # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss
    
    
class mean_squared_error_for_CRE_TSCAE(nn.Module):
    """
    Custom MSE loss that only computes error at positions where y_true != 0.
    This is typically used in reconstruction tasks where some target values are masked (set to 0).
    """

    def __init__(self):
        super(mean_squared_error_for_CRE_TSCAE, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): predicted values, shape (batch, ...)
            y_true (Tensor): ground truth values, same shape as y_pred

        Returns:
            Tensor: scalar loss value (mean squared error over non-zero y_true positions)
        """
        mask = (y_true != 0).float()                        # 1 for valid positions, 0 for masked
        squared_error = (y_pred - y_true) ** 2
        masked_squared_error = squared_error * mask
        loss = masked_squared_error.sum() / (mask.sum() + 1e-8)  # avoid division by zero
        return loss
    

class TripletLoss_for_CRE_TSCAE(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss_for_CRE_TSCAE, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # Normalize for stability
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        pairwise_dist = self._pairwise_distances(embeddings)

        labels = labels.unsqueeze(1)
        label_equal = (labels == labels.T).float()
        label_not_equal = 1.0 - label_equal

        mask_anchor_positive = label_equal - torch.eye(labels.size(0), device=labels.device)
        mask_anchor_negative = label_not_equal

        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)

        max_anchor_negative_dist = pairwise_dist.max(dim=1, keepdim=True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)

        loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        if torch.isnan(loss.mean()):
            print("⚠️ NaN in triplet loss")
        return loss.mean()

    def _pairwise_distances(self, embeddings):
        dot_product = torch.matmul(embeddings, embeddings.T)
        square_norm = torch.diagonal(dot_product)
        distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        distances = torch.clamp(distances, min=1e-12)  # 防止负数
        return distances



class LMMD_for_AdaptEEG(nn.Module):
    def __init__(self, num_classes=2, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_for_AdaptEEG, self).__init__()
        self.num_classes = num_classes
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # shape: [n_total, n_total]

    def forward(self, f_s, f_t, y_s, p_t):
        batch_size_s = f_s.size(0)
        batch_size_t = f_t.size(0)
        features = torch.cat([f_s, f_t], dim=0)  # [N_s + N_t, D]
        kernels = self.gaussian_kernel(f_s, f_t)

        loss = torch.tensor(0.0, device=f_s.device)
        for k in range(self.num_classes):
            # 源域类别权重
            mask_s = (y_s == k).float()  # [N_s]
            count_s = torch.sum(mask_s)
            if count_s.item() == 0:
                continue
            w_s = mask_s / count_s  # [N_s]

            # 目标域类别权重（伪标签）
            mask_t = (p_t == k).float()  # [N_t]
            count_t = torch.sum(mask_t)
            if count_t.item() == 0:
                continue
            w_t = mask_t / count_t  # [N_t]

            # 计算权重外积
            W_ss = torch.matmul(w_s.view(-1, 1), w_s.view(1, -1))  # [N_s, N_s]
            W_tt = torch.matmul(w_t.view(-1, 1), w_t.view(1, -1))  # [N_t, N_t]
            W_st = torch.matmul(w_s.view(-1, 1), w_t.view(1, -1))  # [N_s, N_t]

            # 从核矩阵中提取对应的块
            K_ss = kernels[:batch_size_s, :batch_size_s]
            K_tt = kernels[batch_size_s:, batch_size_s:]
            K_st = kernels[:batch_size_s, batch_size_s:]

            loss += (
                torch.sum(W_ss * K_ss)
                + torch.sum(W_tt * K_tt)
                - 2 * torch.sum(W_st * K_st)
            )

        return loss


class CCL_for_AdaptEEG(nn.Module):
    def __init__(self, num_classes=2, eps=1e-8):
        super(CCL_for_AdaptEEG, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, class_probs):
        """
        class_probs: [B, K], softmax probability of classifier on target samples
        """
        B, K = class_probs.size()
        # Step 1: Compute entropy of each sample
        entropy = -torch.sum(class_probs * torch.log(class_probs + self.eps), dim=1)  # [B]

        # Step 2: Compute normalized weight for each sample
        weights = 1 + torch.exp(-entropy)  # [B]
        weights = B * weights / torch.sum(weights + self.eps)  # sum(weights) = B

        # Step 3: Compute weighted class probabilities
        # Broadcast weights along class dim: [B, 1] * [B, K] => [B, K]
        weighted_probs = class_probs * weights.unsqueeze(1)

        # Step 4: Compute class confusion matrix (CCM) = Y^T (Y ∘ a)
        CCM = torch.matmul(class_probs.T, weighted_probs)  # [K, K]

        # Step 5: Row-normalize CCM to get CCM'
        row_sums = torch.sum(CCM, dim=1, keepdim=True) + self.eps  # [K, 1]
        CCM_normalized = CCM / row_sums  # [K, K]

        # Step 6: Compute CCL Loss
        total_sum = torch.sum(CCM_normalized)  # sum of all elements
        diag_sum = torch.trace(CCM_normalized)  # sum of diagonal elements
        loss = (total_sum - diag_sum) / self.num_classes  # Eq. (12)

        return loss



class mmd_loss_for_SDDA(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(mmd_loss_for_SDDA, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.epsilon = 1e-8  # 防止除以0

    def gaussian_kernel(self, x, y):
        """
        x: (n_x, d)
        y: (n_y, d)
        return: (n_x, n_y) kernel matrix
        """
        # 确保输入没有 NaN 或 Inf
        if torch.isnan(x).any() or torch.isinf(x).any() or torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: input to gaussian_kernel contains NaN or Inf.")
            return torch.zeros(x.size(0), y.size(0)).to(x.device)

        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # (n_x, 1)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)  # (n_y, 1)

        dist = x_norm + y_norm.T - 2 * torch.mm(x, y.T)  # (n_x, n_y)
        dist = torch.clamp(dist, min=0.0)  # 防止负数带来 sqrt 或 exp 问题

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.mean(dist).detach() + self.epsilon  # 加 epsilon 防止除以0

        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-dist / (bw + self.epsilon)) for bw in bandwidth_list]

        return sum(kernel_val)  # (n_x, n_y)

    def forward(self, source, target):
        if source.size(0) == 0 or target.size(0) == 0:
            print("Warning: source or target is empty in MMD loss.")
            return torch.tensor(0.0, requires_grad=True).to(source.device)

        batch_size_s = source.size(0)
        batch_size_t = target.size(0)

        K_ss = self.gaussian_kernel(source, source)  # (Ns, Ns)
        K_tt = self.gaussian_kernel(target, target)  # (Nt, Nt)
        K_st = self.gaussian_kernel(source, target)  # (Ns, Nt)

        loss = (
            K_ss.sum() / (batch_size_s ** 2 + self.epsilon) +
            K_tt.sum() / (batch_size_t ** 2 + self.epsilon) -
            2 * K_st.sum() / (batch_size_s * batch_size_t + self.epsilon)
        )

        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: MMD loss is NaN or Inf, returning 0.")
            return torch.tensor(0.0, requires_grad=True).to(source.device)

        return loss


class cosine_center_loss_loss_for_SDDA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, labels):
        """
        features: (B, D) - 特征向量
        labels: (B,)     - 对应的类别标签
        """
        features = F.normalize(features, dim=1)  # 单位向量归一化
        unique_labels = torch.unique(labels)

        centers = []
        for cls in unique_labels:
            mask = (labels == cls)
            cls_features = features[mask]
            cls_center = F.normalize(cls_features.mean(dim=0, keepdim=True), dim=1)  # 单位向量
            centers.append(cls_center)

        centers = torch.cat(centers, dim=0)  # shape: (num_classes_in_batch, D)

        # 为每个样本找到对应的center
        label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
        center_indices = torch.tensor([label_to_index[l.item()] for l in labels], device=features.device)
        selected_centers = centers[center_indices]  # shape: (B, D)

        cosine_sim = (features * selected_centers).sum(dim=1)  # shape: (B,)
        loss = 1 - cosine_sim.mean()

        return loss


class CorrelationAlignmentLoss_for_SCLDGN(nn.Module):
    r"""The `Correlation Alignment Loss` in
    `Deep CORAL: Correlation Alignment for Deep Domain Adaptation (ECCV 2016) <https://arxiv.org/pdf/1607.01719.pdf>`_.

    Given source features :math:`f_S` and target features :math:`f_T`, the covariance matrices are given by

    .. math::
        C_S = \frac{1}{n_S-1}(f_S^Tf_S-\frac{1}{n_S}(\textbf{1}^Tf_S)^T(\textbf{1}^Tf_S))
    .. math::
        C_T = \frac{1}{n_T-1}(f_T^Tf_T-\frac{1}{n_T}(\textbf{1}^Tf_T)^T(\textbf{1}^Tf_T))

    where :math:`\textbf{1}` denotes a column vector with all elements equal to 1, :math:`n_S, n_T` denotes number of
    source and target samples, respectively. We use :math:`d` to denote feature dimension, use
    :math:`{\Vert\cdot\Vert}^2_F` to denote the squared matrix `Frobenius norm`. The correlation alignment loss is
    given by

    .. math::
        l_{CORAL} = \frac{1}{4d^2}\Vert C_S-C_T \Vert^2_F

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, d)` where d means the dimension of input features, :math:`N=n_S=n_T` is mini-batch size.
        - Outputs: scalar.
    """

    def __init__(self):
        super().__init__()

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)

        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()

        return mean_diff + cov_diff
