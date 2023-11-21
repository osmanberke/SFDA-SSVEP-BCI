# Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces
This is the official repository for the paper titled "Source-Free Domain Adaptation for SSVEP-based Brain-Computer Interfaces" (Arxiv link: https://arxiv.org/abs/2305.17403). This repository allows you to generate and adapt pre-trained DNNs. The pre-trained DNNs are generated in "pre_train.m" and those generated DNNs are adapted in "main.m".

# Preparation
The Benchmark dataset [2] and BETA dataset [3] must be downloaded. The link for the both datasets: http://bci.med.tsinghua.edu.cn/download.html.

# Evaluating the proposed source-free domain adaptation method
In our performance evaluations, we conducted the comparisons (following the procedure in the literature) in a leave-one-participant-out fashion.
For example, we generate the pre-trained DNN using data from 34 (69) participants and adapt the generated DNN in source-free fashion on the remaining test participant, who is considered a new user, using only unlabeled data of the remaining test participant. We get initial predictions from either the pre-trained initial model or the FBCCA method and we choose the one having an initial higher silhouette score (for the details we kindly refer readers to [1]). The predictions of FBCCA are generated in "fbcca_classification.m". 
After adapting the DNN, we calculate the accuracy and ITR performance with true labels at the end of the adaptation. This process is repeated 35 (70) times in the case of the benchmark (BETA) dataset. While calculating the information transfer rate (ITR) results, a 0.5 second gaze shift time is taken into account. We use the DNN architecture of [4] as a DNN architecture, where we use three sub-bands and nine channels (Pz, PO3, PO5, PO4, PO6, POz, O1, Oz, O2).

# References 
1. Osman Berke Guney, Deniz Kucukahmetler, and Huseyin Ozkan, "Source Free Domain Adaptation of a DNN for SSVEP-based Brain-Computer Interfaces", arXiv, 2023.
2. Y. Wang, X. Chen, X. Gao, and S. Gao, “A benchmark dataset for
   ssvep-based brain–computer interfaces,” IEEE Transactions on Neural Systems and 
   Rehabilitation Engineering,vol. 25, no. 10, pp. 1746–1752, 2016.
3. B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
   benchmark database toward ssvep-bci application,” Frontiers in
   Neuroscience, vol. 14, p. 627, 2020.
4. O. B. Guney, M. Oblokulov and H. Ozkan, "A Deep Neural Network for SSVEP-Based Brain-Computer Interfaces," IEEE Transactions on Biomedical Engineering, vol. 69, no. 2, pp. 932-944,  2022.
