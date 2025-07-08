import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable
from detectron2.utils.events import get_event_storage

from .build import KD_REGISTRY
from .base import RCNNKD
from .utils import kl_div
from .reviewkd import hcl, build_abfs
from .gdkd3 import gdkd3_loss


@KD_REGISTRY.register()
class GDKDS(RCNNKD):
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

        losses["loss_gdkds"], info_dict = rcnn_gdkd_s_loss(
            s_predictions,
            t_predictions,
            [x.gt_classes for x in sampled_proposals],
            w0=self.kd_args.GDKDS.W0,
            w1=self.kd_args.GDKDS.W1,
            w2=self.kd_args.GDKDS.W2,
            w3=self.kd_args.GDKDS.W3,
            temperature=self.kd_args.GDKDS.T,
            bg_src=self.kd_args.GDKDS.BG_SRC,
            bg_distill_type=self.kd_args.GDKDS.BG_DISTILL_TYPE,
        )

        if self.kd_args.GDKDS.WARMUP > 0:
            storage = get_event_storage()
            losses["loss_gdkds"] = min(storage.iter/self.kd_args.GDKDS.WARMUP, 1.0) * losses["loss_gdkds"]
        self.record_info(info_dict)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

@KD_REGISTRY.register()
class ReviewGDKDS(RCNNKD):
    @configurable
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        kd_args,
    ):
        super().__init__(student=student, teacher=teacher, kd_args=kd_args)

        self.abfs = build_abfs(
            in_channels=kd_args.REVIEWKD.IN_CHANNELS,
            out_channels=kd_args.REVIEWKD.OUT_CHANNELS,
            mid_channel=kd_args.REVIEWKD.MAX_MID_CHANNEL
        )

    def _forward_pure_roi_head(self, roi_head, features, proposals):
        features = [features[f] for f in roi_head.box_in_features]
        box_features = roi_head.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = roi_head.box_head(box_features)
        predictions = roi_head.box_predictor(box_features)
        return predictions

    def _forward_abf_trans(self, student_features):
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(features, res_features)
            results.insert(0, out_features)

        return results

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        s_images = self.preprocess_image(batched_inputs)
        t_images = self.preprocess_image_teacher(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
        else:
            gt_instances = None

        s_features = self.student.backbone(s_images.tensor)
        t_features = self.teacher.backbone(t_images.tensor)

        losses = {}
        if self.student.proposal_generator is not None:
            proposals, proposal_losses = self.student.proposal_generator(
                s_images, s_features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device)
                         for x in batched_inputs]
            proposal_losses = {}

        sampled_proposals, detector_losses = self.student.roi_heads(
            s_images, s_features, proposals, gt_instances)

        # TODO: avoid duplicate forward for student
        s_predictions = self._forward_pure_roi_head(
            self.student.roi_heads, s_features, sampled_proposals)
        t_predictions = self._forward_pure_roi_head(
            self.teacher.roi_heads, t_features, sampled_proposals)

        losses["loss_gdkds"], info_dict = rcnn_gdkd_s_loss(
            s_predictions,
            t_predictions,
            [x.gt_classes for x in sampled_proposals],
            w0=self.kd_args.GDKDS.W0,
            w1=self.kd_args.GDKDS.W1,
            w2=self.kd_args.GDKDS.W2,
            w3=self.kd_args.GDKDS.W3,
            temperature=self.kd_args.GDKDS.T,
            bg_src=self.kd_args.GDKDS.BG_SRC,
            bg_distill_type=self.kd_args.GDKDS.BG_DISTILL_TYPE,
        )

        if self.kd_args.GDKDS.WARMUP > 0:
            storage = get_event_storage()
            losses["loss_gdkds"] = min(storage.iter/self.kd_args.GDKDS.WARMUP, 1.0) * losses["loss_gdkds"]

        self.record_info(info_dict)

        s_features_flat = [s_features[f] for f in s_features]
        t_features_flat = [t_features[f] for f in t_features]

        s_features_flat_trans = self._forward_abf_trans(s_features_flat)
        losses['loss_reviewkd'] = hcl(
            s_features_flat_trans, t_features_flat) * self.kd_args.REVIEWKD.KD_WEIGHT

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

def rcnn_gdkd_s_loss(
    s_predictions,
    t_predictions,
    gt_classes,
    w0,
    w1,
    w2,
    w3,
    temperature,
    bg_src="target",
    bg_distill_type="dkd",
):
    "GDKD3 with seperate handling of background logits: GDKD(k=1) or DKD"

    s_logits, s_bbox_offsets = s_predictions
    t_logits, t_bbox_offsets = t_predictions
    gt_classes = torch.cat(tuple(gt_classes), 0).reshape(-1)

    info_dict = {}

    bg_class_ind = s_logits.shape[1] - 1

    if bg_src == "teacher":
        bg_mask = torch.argmax(t_logits, dim=1) == bg_class_ind
    elif bg_src == "target":
        bg_mask = gt_classes == bg_class_ind
    else:
        raise ValueError(f"Unknown background source: {bg_src}")
    fg_mask = ~bg_mask

    fg_loss_gdkd, fg_loss_info_dict = gdkd3_loss(
        s_logits[fg_mask], t_logits[fg_mask], w0, w1, temperature
    )

    if bg_distill_type == "dkd":
        target = gt_classes[bg_mask]
    elif bg_distill_type == "gdkd":
        target = None  # use top1
    else:
        raise ValueError(f"Unknown background distill type: {bg_distill_type}")

    bg_loss_gdkd, bg_loss_info_dict = dkd_loss(
        s_logits[bg_mask],
        t_logits[bg_mask],
        alpha=w2,
        beta=w3,
        temperature=temperature,
        target=target,
    )
    bg_loss_info_dict = {f"bg_{k}": v for k, v in bg_loss_info_dict.items()}

    loss_gdkd = fg_loss_gdkd + bg_loss_gdkd

    info_dict.update(fg_loss_info_dict)
    info_dict.update(bg_loss_info_dict)

    return loss_gdkd, info_dict


def get_top1_masks(logits):
    # NOTE: masks are calculated in cuda

    # top1 mask
    max_indices = logits.argmax(dim=1, keepdim=True)
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, max_indices, 1)

    # other mask
    mask_u2 = torch.ones_like(logits, dtype=torch.bool).scatter_(1, max_indices, 0)

    return mask_u1, mask_u2


def get_target_masks(logits, target):
    # NOTE: masks are calculated in cuda

    target = target.reshape(-1)
    mask_u1 = torch.zeros_like(logits, dtype=torch.bool).scatter_(
        1, target.unsqueeze(1), 1
    )
    mask_u2 = torch.ones_like(logits, dtype=torch.bool).scatter_(
        1, target.unsqueeze(1), 0
    )

    return mask_u1, mask_u2


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)  # [B, 2]
    return rt


def dkd_loss(
    logits_student,
    logits_teacher,
    alpha,
    beta,
    temperature,
    target=None,
    mask_magnitude=1000,
    kl_type="forward",
):
    if target is None:
        gt_mask, other_mask = get_top1_masks(logits_teacher)
    else:
        gt_mask, other_mask = get_target_masks(logits_teacher, target)

    soft_logits_student = logits_student / temperature
    soft_logits_teacher = logits_teacher / temperature

    p_student = F.softmax(soft_logits_student, dim=1)
    p_teacher = F.softmax(soft_logits_teacher, dim=1)
    p0_student = cat_mask(p_student, gt_mask, other_mask)
    p0_teacher = cat_mask(p_teacher, gt_mask, other_mask)

    log_p0_student = torch.log(p0_student)
    tckd_loss = F.kl_div(log_p0_student, p0_teacher, reduction="batchmean") * (
        temperature**2
    )

    log_p2_student = F.log_softmax(
        soft_logits_student - mask_magnitude * gt_mask, dim=1
    )
    log_p2_teacher = F.log_softmax(
        soft_logits_teacher - mask_magnitude * gt_mask, dim=1
    )
    nckd_loss = kl_div(log_p2_student, log_p2_teacher, temperature, kl_type=kl_type)

    dkd_loss = alpha * tckd_loss + beta * nckd_loss

    info = dict(
        loss_tckd=tckd_loss.detach(),
        loss_nckd=nckd_loss.detach(),
    )

    return dkd_loss, info
