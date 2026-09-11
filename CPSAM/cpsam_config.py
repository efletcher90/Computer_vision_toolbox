from unet_config import PATH_START


CPSAM_MODEL_FILE = None
# CPSAM_MODEL_FILE = str(PATH_START / pl.Path(r"\src\saved_models\cpsam_onl_org_count_v1"))

CPSAM_FLOW_THRESHOLD = 0.4

# Fraction of a CPSAM cell instance's own area that must fall inside the
# region CNN's mask for that cell to be kept as "in the ONL".
MIN_OVERLAP_FRACTION = 0.25

# Maximum area (px^2, at CPSAM/original resolution) of an enclosed hole that
# gets filled during region mask cleanup. Plain hole-filling can't tell a
# small gap between neighbouring cells apart from a genuine large lumen
# inside a ring-shaped band. This caps it so only small
# gaps get filled and large enclosed regions stay open.
MAX_HOLE_AREA = 5000

