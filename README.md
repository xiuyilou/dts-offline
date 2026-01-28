# **Decoding Tree Sketching (DTS)**

## 💡 About
Large Reasoning Models (LRMs) achieve remarkable inference-time improvements through parallel thinking. However, existing methods rely on
redundant sampling of reasoning trajectories, failing to effectively explore the reasoning space to
uncover high-quality solutions. To address these
limitations, we propose Decoding Tree Sketch-
ing (DTS), a plug-and-play decoding framework
for structural multi-trajectory exploration and reasoning selection. For reasoning exploration, DTS
sketches a backbone tree of the reasoning space
by selectively branching at decision tokens. For
reasoning selection, guided by length-accuracy
anti-correlation, DTS designs an early termination to prioritize short and reliable trajectories
during decoding. Experimental results across
four LRMs and datasets demonstrate that DTS
significantly enhances accuracy by 14% and reduces repetitive generation by 8% on average.
Notably, DTS enables smaller models to outperform larger models with 10× the size, highlighting its potential to strengthen reasoning capabilities.










