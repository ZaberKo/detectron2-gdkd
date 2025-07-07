from detectron2.config import get_cfg, CfgNode as CN, LazyCall as L


base_cfg = get_cfg()

base_cfg.MODEL.MOBILENETV2 = CN()
base_cfg.MODEL.MOBILENETV2.DEBUG = 0
base_cfg.MODEL.MOBILENETV2.OUT_FEATURES = ['m2']
base_cfg.MODEL.MOBILENETV2.NORM = 'FrozenBN'

# model_cfg = base_cfg.MODEL.clone()
# input_cfg = base_cfg.INPUT.clone()

cfg = base_cfg.clone()
cfg.TEACHER = base_cfg.clone()

# cfg.TEACHER=CN()
# cfg.TEACHER.MODEL = model_cfg.clone()
# cfg.TEACHER.INPUT = input_cfg.clone()
# cfg.STUDENT = CN()
# cfg.STUDENT.MODEL = model_cfg.clone()
# cfg.STUDENT.INPUT = input_cfg.clone()


cfg.EXPERIMENT = CN()
cfg.EXPERIMENT.PROJECT = "detection_coco"
cfg.EXPERIMENT.NAME = ""
cfg.EXPERIMENT.TAG = []
cfg.EXPERIMENT.WANDB = True

cfg.KD = CN()
cfg.KD.TYPE = "DKD"

cfg.KD.REVIEWKD = CN()
cfg.KD.REVIEWKD.KD_WEIGHT = 1.0
# ABF settings:
cfg.KD.REVIEWKD.IN_CHANNELS = [256, 256, 256, 256, 256]
cfg.KD.REVIEWKD.OUT_CHANNELS = [256, 256, 256, 256, 256]
cfg.KD.REVIEWKD.MAX_MID_CHANNEL = 256

cfg.KD.DKD = CN()
cfg.KD.DKD.ALPHA = 1.0
cfg.KD.DKD.BETA = 0.25
cfg.KD.DKD.T = 1.0
cfg.KD.DKD.DISTILL_TYPE =  "all" # all or fg

cfg.KD.GDKD = CN()
cfg.KD.GDKD.TOPK = 5
cfg.KD.GDKD.W0 = 1.0
cfg.KD.GDKD.W1 = 0.125
cfg.KD.GDKD.W2 = 0.25
cfg.KD.GDKD.T = 1.0
cfg.KD.GDKD.DISTILL_TYPE = "all" # all or fg
# cfg.KD.GDKD.WARMUP = 36000 # unit: iters

cfg.KD.GDKD3 = CN()
cfg.KD.GDKD3.W0 = 1.0
cfg.KD.GDKD3.W1 = 0.25
cfg.KD.GDKD3.T = 1.0
cfg.KD.GDKD3.DISTILL_TYPE = "all" # all or fg


cfg.KD.GDKDS = CN()
cfg.KD.GDKDS.TOPK = 2
cfg.KD.GDKDS.W0 = 1.0
cfg.KD.GDKDS.W1 = 0.125
cfg.KD.GDKDS.W2 = 0.25
cfg.KD.GDKDS.W3 = 1.0 # DKD alpha for bg
cfg.KD.GDKDS.W4 = 0.25 # DKD beta for bg
cfg.KD.GDKDS.BG_SRC = "teacher"  # "teacher" or "target"
cfg.KD.GDKDS.BG_DISTILL_TYPE = "gdkd"  # "dkd" or "gdkd"
cfg.KD.GDKDS.WARMUP = 5000  # unit: iters
cfg.KD.GDKDS.T = 1.0

def get_distiller_config():
    return cfg.clone()

    