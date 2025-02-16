import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage

from .build import KD_REGISTRY
from .base import RCNNKD
from .utils import kl_div


@KD_REGISTRY.register()
class GDKD(RCNNKD):
    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        kd_args,
    ):
        super().__init__(student=student, teacher=teacher, kd_args=kd_args)

    def _forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return predictions

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        s_images = self.preprocess_image(batched_inputs)
        t_images = self.preprocess_image_teacher(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        s_features = self.student.backbone(s_images.tensor)
        t_features = self.teacher.backbone(t_images.tensor)

        losses = {}
        if self.student.proposal_generator is not None:
            proposals, proposal_losses = self.student.proposal_generator(
                s_images, s_features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.student.roi_heads(
            s_images, s_features, proposals, gt_instances
        )

        # TODO: avoid duplicate forward for student
        s_predictions = self._forward_pure_roi_head(
            self.student.roi_heads, s_features, sampled_proposals
        )
        t_predictions = self._forward_pure_roi_head(
            self.teacher.roi_heads, t_features, sampled_proposals
        )

        losses["loss_gdkd"], info_dict = rcnn_gdkd_loss(
            s_predictions,
            t_predictions,
            [x.gt_classes for x in sampled_proposals],
            k=self.kd_args.GDKD.TOPK,
            w0=self.kd_args.GDKD.W0,
            w1=self.kd_args.GDKD.W1,
            w2=self.kd_args.GDKD.W2,
            temperature=self.kd_args.GDKD.T,
            distill_type=self.kd_args.GDKD.DISTILL_TYPE,
        )

        self.record_info(info_dict)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


@KD_REGISTRY.register()
class GDKD3(RCNNKD):
    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        kd_args,
    ):
        super().__init__(student=student, teacher=teacher, kd_args=kd_args)

    def _forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return predictions

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        s_images = self.preprocess_image(batched_inputs)
        t_images = self.preprocess_image_teacher(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        s_features = self.student.backbone(s_images.tensor)
        t_features = self.teacher.backbone(t_images.tensor)

        losses = {}
        if self.student.proposal_generator is not None:
            proposals, proposal_losses = self.student.proposal_generator(
                s_images, s_features, gt_instances
            )
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.student.roi_heads(
            s_images, s_features, proposals, gt_instances
        )

        # TODO: avoid duplicate forward for student
        s_predictions = self._forward_pure_roi_head(
            self.student.roi_heads, s_features, sampled_proposals
        )
        t_predictions = self._forward_pure_roi_head(
            self.teacher.roi_heads, t_features, sampled_proposals
        )

        losses["loss_gdkd3"], info_dict = rcnn_gdkd3_loss(
            s_predictions,
            t_predictions,
            [x.gt_classes for x in sampled_proposals],
            w0=self.kd_args.GDKD3.W0,
            w1=self.kd_args.GDKD3.W1,
            temperature=self.kd_args.GDKD3.T,
            distill_type=self.kd_args.GDKD3.DISTILL_TYPE,
        )

        self.record_info(info_dict)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


def rcnn_gdkd_loss(
    s_predictions, t_predictions, gt_classes, k, w0, w1, w2, temperature, distill_type
):
    s_logits, s_bbox_offsets = s_predictions
    t_logits, t_bbox_offsets = t_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)

    empty_fg_flag = False
    info_dict = {}
    if distill_type == "fg":
        # ignore background images
        num_classes = s_logits.shape[1]
        batch_size = gt_classes.shape[0]
        mask = gt_classes != num_classes - 1
        ratio = mask.sum().item() / batch_size
        info_dict["distill_fg_ratio"] = ratio

        s_logits = s_logits[mask]
        t_logits = t_logits[mask]
        empty_fg_flag = torch.count_nonzero(mask) == 0

    loss_gdkd, loss_info_dict = gdkd_loss(
        s_logits, t_logits, k, w0, w1, w2, temperature
    )

    if empty_fg_flag:
        loss_gdkd = torch.zeros_like(loss_gdkd)
        loss_info_dict = {k: torch.zeros_like(v) for k, v in loss_info_dict.items()}

    info_dict.update(loss_info_dict)

    return loss_gdkd, info_dict


def rcnn_gdkd3_loss(
    s_predictions, t_predictions, gt_classes, w0, w1, temperature, distill_type
):
    s_logits, s_bbox_offsets = s_predictions
    t_logits, t_bbox_offsets = t_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)

    empty_fg_flag = False
    info_dict = {}
    if distill_type == "fg":
        # ignore background images
        num_classes = s_logits.shape[1]
        batch_size = gt_classes.shape[0]
        mask = gt_classes != num_classes - 1
        ratio = mask.sum().item() / batch_size
        info_dict["distill_fg_ratio"] = ratio

        s_logits = s_logits[mask]
        t_logits = t_logits[mask]
        empty_fg_flag = torch.count_nonzero(mask) == 0

    loss_gdkd, loss_info_dict = gdkd3_loss(s_logits, t_logits, w0, w1, temperature)

    if empty_fg_flag:
        loss_gdkd = torch.zeros_like(loss_gdkd)
        loss_info_dict = {k: torch.zeros_like(v) for k, v in loss_info_dict.items()}

    info_dict.update(loss_info_dict)

    return loss_gdkd, info_dict


def get_masks(logits, k=5):
    largest_flag = True

    ranks = torch.topk(logits, k, dim=-1, largest=largest_flag, sorted=False).indices

    # topk mask
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)
    # other mask
    mask_u2 = torch.logical_not(mask_u1)

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def gdkd_loss(
    logits_student,
    logits_teacher,
    k,
    w0,
    w1,
    w2,
    temperature,
    mask_magnitude=1000,
    kl_type="forward",
):
    mask_u1, mask_u2 = get_masks(logits_teacher, k)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask(p_student, mask_u1, mask_u2)
    p0_teacher = cat_mask(p_teacher, mask_u1, mask_u2)

    log_p0_student = torch.log(p0_student)
    high_loss = F.kl_div(log_p0_student, p0_teacher, reduction="batchmean") * (
        temperature**2
    )

    # topk loss
    log_p1_student = F.log_softmax(
        soft_logits_student - mask_magnitude * mask_u2, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * mask_u2, dim=1
    )
    low_top_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    # other classes loss
    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * mask_u1, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * mask_u1, dim=1
    )
    low_other_loss = kl_div(log_p2_student, log_p2_teacher, temperature, kl_type)

    gdkd_loss = w0 * high_loss + w1 * low_top_loss + w2 * low_other_loss

    info = dict(
        loss_high=high_loss.detach(),
        loss_low_top=low_top_loss.detach(),
        loss_low_other=low_other_loss.detach(),
    )

    return gdkd_loss, info


def get_masks3(logits):
    largest_flag = True

    ranks = torch.topk(logits, 2, dim=-1, largest=largest_flag, sorted=True).indices

    # top1 mask (usually target)
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks[:, 0:1], 1)
    # top2 mask
    mask_u2 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks[:, 1:2], 1)

    # other mask

    not_mask_u3 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, ranks, 1)

    mask_u3 = torch.logical_not(not_mask_u3)

    return mask_u1, mask_u2, mask_u3, not_mask_u3


def cat_mask3(t, mask1, mask2, mask3):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    t3 = (t * mask3).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2, t3], dim=1)  # [B, 3]
    return rt


def gdkd3_loss(
    logits_student,
    logits_teacher,
    w0,
    w1,
    temperature,
    mask_magnitude=1000,
    kl_type="forward",
):
    mask_u1, mask_u2, mask_u3, not_mask_u3 = get_masks3(logits_teacher)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)

    # Notation: high_loss: level 0 loss; low_loss: level 1 loss
    # accumulated term
    p0_student = cat_mask3(p_student, mask_u1, mask_u2, mask_u3)
    p0_teacher = cat_mask3(p_teacher, mask_u1, mask_u2, mask_u3)

    log_p0_student = torch.log(p0_student)
    high_loss = F.kl_div(log_p0_student, p0_teacher, reduction="batchmean") * (
        temperature**2
    )
    # other classes loss
    log_p1_student = F.log_softmax(
        soft_logits_student - mask_magnitude * not_mask_u3, dim=1
    )
    log_p1_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * not_mask_u3, dim=1
    )
    low_other_loss = kl_div(log_p1_student, log_p1_teacher, temperature, kl_type)

    gdkd_loss = w0 * high_loss + w1 * low_other_loss

    info = dict(
        loss_high=high_loss.detach(),
        loss_low_other=low_other_loss.detach(),
    )

    return gdkd_loss, info
