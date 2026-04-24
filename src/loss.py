import torch
import torch.nn as nn

def recompute_matched(matched, logits, smoothness=0.0):
    """ Recompute the `matched` matrix if the smoothness value is given.
    """
    if smoothness==0.0:
        return matched, None
    else:
        logits = logits.view(matched.size())
        # XXX Warning: all negative pairs will return weird results
        gt_labels, gt_indices = torch.max(matched, dim=1)
        gt_vals = logits[:, gt_indices].diag()
        pseudo_gt_indices = (logits >= gt_vals.unsqueeze(1))
        new_matched = (gt_labels.unsqueeze(1) * (pseudo_gt_indices))
        _matched = matched.clone()
        _matched[pseudo_gt_indices] = new_matched[pseudo_gt_indices]

        return _matched, torch.sum(pseudo_gt_indices).item() - len(gt_indices)


def contrastive_loss(logits, gt_match):
    return nn.functional.cross_entropy(logits, gt_match)

def clip_loss(similarity, gt_match):
    ab_loss = contrastive_loss(similarity,gt_match)
    ba_loss = contrastive_loss(similarity.t(),gt_match.t())
    return 0.5*ab_loss + 0.5*ba_loss


def compute_loss(similarity,pseudo_match_alpha=0.1):
    gt_match = torch.eye(similarity.size(0)).to(similarity.device)
    loss = clip_loss(similarity, gt_match)
    updated_gt_match, _ = recompute_matched(matched=gt_match, logits=similarity, smoothness=pseudo_match_alpha)
    loss_pseudo = clip_loss(similarity, updated_gt_match)
    loss = loss + pseudo_match_alpha*loss_pseudo
    return loss
