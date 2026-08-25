# All region CNN training runs get grouped under this experiment name in MLflow.
# Helps track training following model alterations and runs.
# To start the user interface, type the following into the terminal: mlflow ui
# Then when loaded, open http://localhost:5000).
MLFLOW_EXPERIMENT_NAME = "organoid_onl_counter"

###### ^ make generic



# ---------------------------------------------------------------------------
# Cellpose Segment Anything Model (CPSAM, cell-instance model)
# ---------------------------------------------------------------------------

# Location of the CPSAM model. Leave as None if the default model is needed. For this work, mass cell detection is good
# enough if the CNN is being used for ONL layer gating afterwards
CPSAM_MODEL_FILE = None
# CPSAM_MODEL_FILE = str(PATH_START / pl.Path(r"\src\saved_models\cpsam_onl_org_count_v1"))

# Diameter in px that CPSAM expects cells to be, at the ORIGINAL image
# resolution (not resized). Set this from your training set's actual cell
# diameter distribution -- leave as None only if you've confirmed auto
# diameter estimation performs reliably on your images.
CPSAM_DIAMETER = None  # auto -- the aggressive crop-based diameter=8 workaround
                       # was specifically compensating for the fine-tuned model's
                       # restricted detection; shouldn't be needed with the base model

# Padding (pixels, in full-resolution coordinates) added around the cleaned
# region mask's bounding box before cropping the raw image for CPSAM. Keeps
# cells that sit right at the region boundary from being clipped.
# NOTE: currently unused -- pipeline.py runs CPSAM on the full image, not a
# crop, now that the base model doesn't need the speed workaround. Left here
# in case cropping is reintroduced later.
CPSAM_CROP_PADDING = 50

CPSAM_FLOW_THRESHOLD = 0.4
CPSAM_CELLPROB_THRESHOLD = 0.0

# Explicit channel index to extract from multi-channel images before
# running CPSAM (0=R, 1=G, 2=B for a standard RGB-ordered array).
# Set to None to pass the raw array through UNMODIFIED (all channels as
# loaded) -- this is the default, since the GUI's channel display sliders
# are very likely display-only and do not filter what gets fed to the
# model, meaning the GUI's full-tissue result was probably produced using
# all 3 raw channels, not an isolated one.
CPSAM_CHANNEL_INDEX = None

# Fraction of a CPSAM cell instance's own area that must fall inside the
# region CNN's mask for that cell to be kept as "in the ONL".
MIN_OVERLAP_FRACTION = 0.25

# Density-based region gate: intersects the region CNN's cleaned prediction
# with an independent "densely packed cells" mask built from CPSAM's own
# detected cells. Corrects for the CNN predicting an over-broad region
# (e.g. covering sparse rosette-interior area) -- a failure mode that
# size/shape cleanup alone can't fix once the over-broad area is fused
# into one dominant blob rather than a separate island.
# DENSITY_GATE_SIGMA: smoothing radius (px) for estimating local cell
# packing density -- should roughly match a few cell diameters.
# DENSITY_GATE_PERCENTILE: pixels above this percentile of the smoothed
# density map count as "densely packed". Tune per dataset: lower keeps
# more area, higher is stricter. Check a few _region_cleaned.png outputs
# after changing this to confirm it isn't cutting into the real band.
DENSITY_GATE_SIGMA = 40
DENSITY_GATE_PERCENTILE = 70

# Position-based gate: among significant connected components remaining
# after keep_largest_component, keeps only the one(s) whose topmost point
# is within MAX_ONL_DEPTH of the single topmost candidate -- discarding a
# comparably-sized but much DEEPER false region (e.g. an inner tissue
# layer), even if that false region is itself densely packed.
#
# This is COMPONENT-level, not per-pixel -- an earlier per-pixel/per-column
# "distance from local surface" version was tested and confirmed fragile:
# a gap in the true top band's detection could cause the per-column
# surface estimate to incorrectly fall through to a deeper false region
# for those specific columns. Comparing whole components avoids this.
#
# REQUIRES CONSISTENT IMAGE ORIENTATION (ONL near the top) -- this is a
# deliberate constraint for this early training phase, not a
# general-purpose solution across arbitrary orientations. If images with
# a different orientation are ever fed through this pipeline, this gate
# will incorrectly exclude the real ONL band -- set USE_POSITION_GATE to
# False if you can't guarantee consistent orientation.
USE_DENSITY_GATE = False
USE_POSITION_GATE = False
MAX_ONL_DEPTH = 200  # px, at CPSAM/original image resolution -- tune per
                     # dataset by checking a few _region_cleaned.png outputs

# Maximum area (px^2, at CPSAM/original resolution) of an enclosed hole that
# gets filled during region mask cleanup. Plain hole-filling can't tell a
# small gap between neighbouring cells apart from a genuine large lumen
# inside a ring-shaped band -- confirmed on real data where a ring-shaped
# band's entire enclosed lumen got filled solid. This caps it so only small
# gaps get filled; large enclosed regions (real lumens) stay open.
# Tune by checking _region_cleaned.png on any image with a ring/loop shape.
MAX_HOLE_AREA = 5000
