# **Decoding Tree Sketching (DTS)**

## 💡 About
Large Reasoning Models (LRMs) achieve remark-
able inference-time improvements through paral-
lel thinking. However, existing methods rely on
redundant sampling of reasoning trajectories, fail-
ing to effectively explore the reasoning space to
uncover high-quality solutions. To address these
limitations, we propose Decoding Tree Sketch-
ing (DTS), a plug-and-play decoding framework
for structural multi-trajectory exploration and rea-
soning selection. For reasoning exploration, DTS
sketches a backbone tree of the reasoning space
by selectively branching at decision tokens. For
reasoning selection, guided by length-accuracy
anti-correlation, DTS designs an early termina-
tion to prioritize short and reliable trajectories
during decoding. Experimental results across
four LRMs and datasets demonstrate that DTS
significantly enhances accuracy by 14% and re-
duces repetitive generation by 8% on average.
Notably, DTS enables smaller models to outper-
form larger models with 10× the size, highlight-
ing its potential to strengthen reasoning capabil-
ities.










