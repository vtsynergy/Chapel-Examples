A Chapel implementation of Edge-connected Jaccard Similarity

In addition to a bespoke edge-centric kernel, this work leverages a hand-translated version of the vertex-centric kernel pipeline from cuGraph (https://github.com/rapidsai/cugraph) licensed under Apache 2.0. Please see NOTICE for a description of which elements were reused and where.

Binary file I/O is compatible with the tools from our previous SYCL efforts [SYCL-Jaccard](https://github.com/vtsynergy/SYCL-Jaccard)

Please Cite:
 **Initial Experiences in Porting A GPU Graph Analysis Workload to Chapel**, Paul Sathre, Atharva Gondhalekar, Wu-chun Feng, In *Proceedings of the 10th Annual Chapel Implementers and Users Workshop (CHIUW)*, Virtual, June 2023. (https://chapel-lang.org/CHIUW/2023/Sathre.pdf)
* For reproducibility, input data can be found here: [CHIUW'23 Input CSR Data](https://chrec.cs.vt.edu/SYCL-Jaccard/HPEC22-Data/index.html)
