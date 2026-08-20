"""
Loads the CPSAM (Cellpose-SAM) ViT-L model for inference. It is a cell instance segmenter
"""

import importlib.metadata
from cellpose import models

# Get the version of the CPSAM model used
try:
    _installed_version = importlib.metadata.version("cellpose")
except importlib.metadata.PackageNotFoundError:
    _installed_version = "UNKNOWN -- not found via importlib.metadata"
print(f"[cpsam_utils] Active cellpose version in this runtime: {_installed_version}")

def load_cpsam_model(model_path, use_gpu=True):
    if model_path:
        print(f"Loading custom trained CPSAM model from {model_path}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
    else:
        print("model_path is None/empty -- loading the BASE/GENERIC CPSAM model "
              "(explicitly requesting 'cpsam' by name, since this cellpose version "
              "requires an explicit model reference rather than defaulting on its own).")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")

    diam_mean = getattr(model, "diam_mean", "ATTRIBUTE NOT FOUND")
    print(f"  model.diam_mean = {diam_mean}")

    if diam_mean == "ATTRIBUTE NOT FOUND":
        print("  diam_mean not found directly on model -- listing all attributes "
              "containing 'diam' to find the actual name in this version:")
        diam_related = [a for a in dir(model) if "diam" in a.lower()]
        for attr_name in diam_related:
            print(f"    model.{attr_name} = {getattr(model, attr_name, '(unreadable)')}")
        if not diam_related:
            print("    (no attributes containing 'diam' found at all)")

    return model


def run_cpsam_batch(model, raw_images, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0):
    """
    Run CPSAM over a list of raw (1024x1024px) images - these will likely always be this resolution.

    Returns a list of integer instance-labelled mask arrays, same order/length
    as raw_images. Each mask is shaped like its corresponding input image.
    """
    instance_masks = []

    for img in raw_images:
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        instance_masks.append(masks)

    return instance_masks