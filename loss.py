import torch
import torch.nn as nn
import MinkowskiEngine as ME
from aleatoric_loss import AleatoricLoss

aleatoric_criterion = AleatoricLoss(is_log_sigma=False, res_loss='l1', nb_samples=10)

def get_loss(end_points):
    end_points['score_label'] = end_points['seal_score_label'] * end_points['wrench_score_label']
    loss = aleatoric_criterion(end_points['score_pred'], end_points['sigma_pred'], end_points['score_label'])
    end_points['loss/overall_loss'] = loss
    return loss, end_points


def compute_score_loss(end_points):
    criterion = nn.SmoothL1Loss(reduction='mean')
    score = end_points['score_pred']
    score_label = end_points['score_label']
    loss = criterion(score.view(-1), score_label)
    end_points['loss/score_loss'] = loss
    return loss, end_points
