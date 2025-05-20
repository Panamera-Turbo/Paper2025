# <center> GraphCogent: Overcoming LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding</center>


This repo contains the code, data, and models for "GraphCogent: Overcoming LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding."

🔗 **GitHub**: [https://anonymous.4open.science/r/GraphCogent/README.md](https://anonymous.4open.science/r/GraphCogent/README.md)  
📜 **Paper**: [Added later]() | 📊 **Benchmark**: [Graph4real]() | 🤖 **Agent**: [Huggingface]() 


**📢 Notice: Ongoing Maintenance**: 
This repository is currently under active development. Core code, execution scripts, and related materials will be uploaded soon.

## Key Features

### 1. GraphCogent Framework
We propose **GraphCogent**, a **collaborative agentic framework** that addresses LLMs' limitations in graph reasoning through:
- **Sensory Module**: Standardizes diverse graph text representations (adjacency lists, symbolic notations, linguistic descriptions) via subgraph sampling.
- **Buffer Module**: Integrates and indexes graph data for efficient retrieval across formats (NetworkX, PyG, NumPy).
- **Execution Module**: Combines **tool calling** (for in-toolset tasks) and **model generation** (for out-toolset tasks) for robust reasoning.

### 2. Graph4real Benchmark
We introduce **Graph4real**, a **real-world graph reasoning benchmark** with:
- **Large-scale graphs** (40–1000+ nodes) from **4 domains** (Web, Social, Transportation, Citation).
- **21 diverse tasks** (Structural Querying, Algorithmic Reasoning, Predictive Modeling).
- **Intent-driven queries** mirroring real-world reasoning scenarios.

### 3. Performance Improvements
GraphCogent achieves:

✅ **98.5% accuracy** on in-toolset tasks (20% improvement over baselines).  
✅ **97.6% accuracy** on **1000-node graphs**, where prior methods fail.  
✅ **30–80% reduction in token consumption** compared to agent-based methods.  
✅ **Robust generalization** across public benchmarks (>90% accuracy).

---



Our entire work flow can be summarized as follows:

<div align="center">
<img src="pics\method.jpg" width="800px">
</div>
