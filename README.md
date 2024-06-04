# Progressive Dual Priori Network for Generalized Breast Tumor Segmentation
Abstract—To promote the generalization ability of breast tumor segmentation models, as well as to improve the segmentation performance for breast tumors with smaller size, low-contrast and irregular shape, we propose a progressive dual priori network (PDPNet) to segment breast tumors
from dynamic enhanced magnetic resonance images (DCEMRI) acquired at different centers. The PDPNet first cropped tumor regions with a coarse-segmentation based localization module, then the breast tumor mask was progressively refined by using the weak semantic priori and cross-scale
correlation prior knowledge. To validate the effectiveness of PDPNet, we compared it with several state-of-the-art methods on multi-center datasets. The results showed that, comparing against the suboptimal method, the DSC and HD95 of PDPNet were improved at least by 5.13% and 7.58%
respectively on multi-center test sets. In addition, through ablations, we demonstrated that the proposed localization module can decrease the influence of normal tissues and therefore improve the generalization ability of the model. The weak semantic priors allow focusing on tumor regions
to avoid missing small tumors and low-contrast tumors. The cross-scale correlation priors are beneficial for promoting the shape-aware ability for irregular tumors. Thus integrating them in a unified framework improved the multicenter breast tumor segmentation performance.

details on https://arxiv.org/abs/2310.13574
