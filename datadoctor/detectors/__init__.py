"""DataDoctor detector modules for Views 1, 2, and 3."""
from datadoctor.detectors.oof_probe import compute_oof_probabilities
from datadoctor.detectors.v1_self_conf import compute_v1_scores
from datadoctor.detectors.v2_knn import compute_v2_scores, build_knn_index
from datadoctor.detectors.v3_cross_probe import compute_v3_scores

__all__ = [
    "compute_oof_probabilities",
    "compute_v1_scores",
    "compute_v2_scores",
    "build_knn_index",
    "compute_v3_scores",
]

