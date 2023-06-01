This work contains code that is a reimplementation of a Jaccard Similarity pipeline that is (C) Nvidia and licensed under Apache 2.0.
The reimplementation has been performed by hand and is entirely contained within CuGraph.chpl
From cuGraph (https://github.com/rapidsai/cugraph) revision 3f13ffcdf (Feb. 17, 2021)
* From `cpp/src/utilities/graph_utils.cuh`
  * The original `fill` wrapper function definition was reimplemented as a Chapel promoted scalar-to-array assignment
* From `cpp/src/link_predicition/jaccard.cu`
  * the "row_sum" kernel has been reimplemented as a Chapel GPU `forall` without support for vertex weights
  * the 3D `jaccard_is` kernel has been reimplemented as a 1D Chapel GPU `forall` with 1D->3D logical reindexing
  * the 1D `jaccard_jw` kernel has been reimplemented as a Chapel GPU `forall`

