from typing import Any, Dict

from models import (
    DDLM_Extractor,
    EEGNet,
    LENet,
    LMDANet,
    OneDCNN_BiLSTM,
    Ours_light,
    Ours_light_with_K,
    PLNet,
    PPNN,
    lightweight_erp,
)


def model_dict() -> Dict[str, Any]:
    """Return the model-name to implementation mapping for paper experiments."""
    return {
        # Paper models
        "DDLM-Extractor": DDLM_Extractor,
        "Ours_light_with_K": Ours_light_with_K,

        # Lightweight EEG baselines
        "EEGNet": EEGNet,
        "PLNet": PLNet,
        "PPNN": PPNN,
        "LMDANet": LMDANet,
        "1DCNN_BiLSTM": OneDCNN_BiLSTM,
        "LENet": LENet,
        "lightweight_erp": lightweight_erp,

        # Channel selection baselines (share DDLM-Extractor backbone)
        "Correlation-based Selection": Ours_light_with_K,
        "Frequency-Domain Weighting": Ours_light_with_K,
        "SparseEA": Ours_light_with_K,
        "Gumbel-Softmax Selector": Ours_light_with_K,
        "Gumbel-Top-K Selector": Ours_light_with_K,
        "OS-Selector w/ Deterministic Top-K": Ours_light_with_K,
        "OS-Selector w/o Orthogonality": Ours_light_with_K,
        "OS-Selector (Ours)": Ours_light_with_K,

        # OS-Selector and controls applied to other backbones
        "EEGNet_OS": EEGNet,
        "PLNet_OS": PLNet,
        "EEGNet_Gumbel": EEGNet,
        "EEGNet_OS_wo_Orth": EEGNet,
        "PLNet_Gumbel": PLNet,
        "PLNet_OS_wo_Orth": PLNet,
    }
