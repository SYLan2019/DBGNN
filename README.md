# DBGNN
A dual branch graph neural network for spatial interpolation in traffic scene

A Dual Branch Graph Neural Network based Spatial Interpolation Method for Traffic Data Inference in Unobserved Locations

Complete traffic data collection is crucial for intelligent transportation system, but due to various factors such as cost, it is not possible to deploy sensors at every location. Spatial interpolation can infer the value of unobserved locations from the data of observed locations, providing fine-grained data measurements to better monitor and control traffic. However, existing methods struggle to accurately model the dynamic spatio-temporal dependencies between traffic locations, resulting in unsatisfactory performance of spatial interpolation for unobserved locations in traffic scene. Therefore, we propose a novel dual branch graph neural network (DBGNN) based on dynamic spatio-temporal correlation representation among traffic nodes for spatial interpolation, which is composed of two branches: the Main Branch and the Auxiliary Branch. The main branch stacks multiple spatio-temporal blocks to represent multi-scale dynamic spatio-temporal features. The auxiliary branch is designed as a shallow network structure to focus more on the details of spatial dependencies between nodes, avoiding the over-smooth problem (i.e., the loss of details) caused by too many graph convolutional layers. Finally, the two branches are fused via a self-attention mechanism to adaptively integrate information from different perspectives. Extensive experiments on four public datasets demonstrate that our DBGNN exceeds the state-of-the-art baselines.

# Notes
The codes will be released as soon as we tidy them up.
