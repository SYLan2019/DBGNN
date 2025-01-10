# DBGNN
A dual branch graph neural network for spatial interpolation in traffic scene

A Dual Branch Graph Neural Network based Spatial Interpolation Method for Traffic Data Inference in Unobserved Locations

Complete traffic data collection is crucial for intelligent transportation system, but due to various factors such as cost, it is not possible to deploy sensors at every location. Using spatial interpolation, the traffic data for unobserved locations can be inferred from the data of observed locations, providing fine-grained measurements for improved traffic monitoring and control.
However, existing methods are limited in modeling the dynamic spatio-temporal dependencies between traffic locations, resulting in unsatisfactory performance of spatial interpolation for unobserved locations in traffic scene. To address this issue, we propose a novel dual branch graph neural network (DBGNN) for spatial interpolation by exploiting dynamic spatio-temporal correlation among traffic nodes. The proposed DBGNN is composed of two branches: the main branch and the auxiliary branch. They are designed to capture the wide-range dynamic spatial correlation and the local detailed spatial diffusion between nodes, respectively. Finally, the two branches are fused via a self-attention mechanism. Extensive experiments on six public datasets demonstrate the advantages of our DBGNN over the state-of-the-art baselines.

# Notes
Accepted by Information Fusion in September,2024.

# [BibTeX] [[DOI](https://doi.org/10.1016/j.inffus.2024.102703)]
@article{journals/inffus/ZhuZLWHRP25,

	title = {A dual branch graph neural network based spatial interpolation method for traffic data inference in unobserved locations.},
 
	year = {2025},
 
	journal = {Inf. Fusion},
 
	author = {{Wujiang Zhu} and {Xinyuan Zhou} and {Shiyong Lan} and {Wenwu Wang 001} and {Zhiang Hou} and {Yao Ren} and {Tianyi Pan}}
}
