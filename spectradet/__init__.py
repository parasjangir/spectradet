"""SpectraDet — a lightweight, FFT-enhanced detector for objects in multi-degraded images.

Course project (CS776): *Object Degradation in Multi-degraded Images*.
A compact, anchor-free detector (~2.5M params) built entirely from scratch in PyTorch:
  * a custom frequency-domain (FFT) module that makes the network degradation-aware,
  * SimOTA dynamic label assignment,
  * an IoU-guided detection loss.
"""

__version__ = "0.1.0"
