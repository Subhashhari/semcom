"""ADJSCC-Q: SNR-adaptive, constellation-constrained deep joint source-channel coding.

Composes the ADJSCC attention-feature module (SNR conditioning) with the DeepJSCC-Q
soft-to-hard quantiser (finite M-QAM channel input). See README.md for the 2x2 ablation.
"""

__all__ = ["config", "gdn", "modules", "constellation", "channel", "models", "data"]
