# <center> GraphCogent: Overcoming LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding</center>


This repo contains the code, data, and models for "GraphCogent: Overcoming LLMs’ Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding."

🔗 **GitHub**: [https://anonymous.4open.science/r/GraphCogent](https://anonymous.4open.science/r/GraphCogent)  
📜 **Paper**: [Added later]() | 📊 **Benchmark**: [Graph4real](https://huggingface.co/5SAGI/NIPS2025/tree/main) | 🤖 **Agent**: [Huggingface](https://huggingface.co/5SAGI/NIPS2025/tree/main) 


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

**Overview of GraphCogent:** Sensory Module (left) standardizes various graph text representations through subgraph sampling and conversion; Buffer Module (center) establishes cross-format data (e.g., NetworkX) integrating and indexing transformations; Execution Module (right) enables two reasoning modes: Reasoning Agent is employed for task discrimination and implements tool calling for in-toolset tasks, Model Agent handles out-toolset tasks based on model generation.


## Getting Started

<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Environment Preparation'>1. Environment Preparation </a>
* <a href='#Graph4real Construction Workflow'>2. Graph4real Construction Workflow </a>
  * <a href='#Dataset Preparation Rules'>2.1 Dataset Preparation Rules
  * <a href='#Pipeline Implementation'>2.2 Pipeline Implementation
  * <a href='#Task Scenarios'>2.3 Task Scenarios
  * <a href='#Benchmark Details'>2.4 Benchmark Details
  * <a href='#Execution Example'>2.5 Execution Example

* <a href='#Evaluating GraphCogent'>3. Evaluating GraphCogent</a>
  * <a href='#Task Execution'>3.1 Task Execution</a>
  * <a href='#Task Evaluation'>3.2 Task Evaluation</a>
* <a href='#GraphCogent Training Process'>4. GraphCogent Training Process </a>
  * <a href='#LLaMA-Factory Installation'>4.1. LLaMA-Factory Installation</a>
  * <a href='#Thinking Path Collection'>4.2. Thinking Path Collection</a>
  * <a href='#LoRA Tuning'>4.3. LoRA Tuning</a>
  * <a href='#DPO Tuning'>4.4 DPO Tuning</a>
  * <a href='#Export Model'>4.5 Export Model</a>



****


<span id='Environment Preparation'/>


### 1. Environment Preparation  <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```shell
conda create -n GraphCogent python=3.12

conda activate GraphCogent

# Torch with CUDA 12.3
# Please clone our GraphCogent first, and switch to the directory.
cd GraphCogent
# Install required libraries
pip install -r requirements.txt
```

<span id='Graph4real Construction Workflow'/>

### 2. Graph4real Construction Workflow  <a href='#all_catelogue'>[Back to Top]</a>


<span id='Dataset Preparation Rules'/>

#### 2.1 Dataset Preparation Rules <a href='#all_catelogue'>[Back to Top]</a>

##### Real-World Graph Scaling
- **Domains**: 
  - Web (Google Web Graph)
  - Social (SNAP Social Circles) 
  - Transportation (PeMS)
  - Citation (Cora)
- **Scales**:
  - Small: 40 nodes
  - Medium: 100 nodes
  - Large: 1000 nodes
- **Sampling Method**: Biased random walks preserving topological properties

##### Text Representations
| Format | Example | Use Case |
|--------|---------|----------|
| Adjacency List | `[0,1],[0,2]` | Algorithmic tasks |
| Symbolic Notation | `0→1` | Structural queries |
| Linguistic Descriptions | "Station A connects to Station B" | Domain-specific tasks |

##### Task Design Principles
- **Balance**: 50% true/false for binary tasks
- **Uniqueness**: Single valid solution for algorithmic tasks
- **Diversity**: 5+ prompt templates per task

<span id='Pipeline Implementation'/>

#### 2.2 Pipeline Implementation <a href='#all_catelogue'>[Back to Top]</a>

##### Directory Structure

```
Graph4real/
├── generation/
│   ├── sampling.py          # Graph sampling
│   ├── task_generator.py    # Question generation
│   └── validate.py          # Quality checks
├── data/
│   ├── raw/                 # Source datasets
│   └── processed/           # Sampled graphs
└── tasks/                   # Final benchmark
```


##### Key Scripts

##### Graph Sampling
```bash
python generation/sampling.py \
  --input_path data/raw/transportation/PeMS.edgelist \
  --output_dir data/processed/transportation/100_nodes \
  --scale 100 \
  --walk_length 30
```


##### Task Generation
```bash
python generation/task_generator.py \
  --graph_data data/processed/web/100_nodes \
  --template prompts/pagerank.txt \
  --output tasks/web/pagerank.json \
  --num_examples 200
```

<span id='Task Scenarios'/>

#### 2.3 Task Scenarios <a href='#all_catelogue'>[Back to Top]</a>

| **Domain**     | **Scenarios** |
|----------------|---------------|
| **Social**     | - Information diffusion analysis <br> - Community detection and recommendation systems <br> - Fraudulent account detection <br> - Influence maximization algorithms <br> - Social network dynamics analysis |
| **Web**        | - Web crawler efficiency optimization <br> - Search engine ranking optimization <br> - Web structural integrity diagnosis <br> - Topical community discovery <br> - DDoS attack mitigation |
| **Transportation** | - Travel route planning <br> - Logistics delivery optimization <br> - Urban emergency response planning <br> - Public transit network scheduling <br> - Shared mobility platforms |
| **Citation**   | - Scholarly influence tracking <br> - Interdisciplinary research identification <br> - Seminal paper discovery <br> - Literature retrieval ranking optimization <br> - Research frontier identification |

<span id='Benchmark Details'/>

#### 2.4 Benchmark Details <a href='#all_catelogue'>[Back to Top]</a>

| Task              | Description                                                                 | Tool Algorithm     | Time Complexity       |
|-------------------|-----------------------------------------------------------------------------|--------------------|-----------------------|
| Edge Count        | Count the total number of edges in a given graph.                          | Direct Lookup      | $O(1)$               |
| Node Count        | Count the total number of nodes in a given graph.                          | Direct Lookup      | $O(1)$               |
| Degree Count      | Count the number of edges connected to a specific node in a given graph.   | Direct Lookup      | $O(1)$               |
| Edge Existence    | Determine whether a specific edge exists between two nodes in a given graph. | Direct Lookup      | $O(1)$               |
| Node Existence    | Determine whether a specific node exists in a given graph.                 | Direct Lookup      | $O(1)$               |
| Cycle Detection   | Determine whether there exists any cycle in a given graph.                 | Depth-First Search | $O(\|V\|+\|E\|)$     |
| Triangle Count    | Count the total number of triangles in a given graph.                      | Node Iterator      | $O(\|V\| \cdot d_{\text{max}}^2)$ |
| Path Existence    | Determine whether a specific path exists between two nodes in a given graph. | Depth-First Search | $O(\|V\|+\|E\|)$     |
| Shortest Path     | Determine the minimum distance between two nodes in a given graph.        | Dijkstra           | $O({\|V\|}^2+\|E\|)$ |

> **Total**: 4,200 questions across 21 tasks

<span id='Execution Example'/>

#### 2.5 Execution Example <a href='#all_catelogue'>[Back to Top]</a>

```bash
# Generate full Transportation benchmark
bash scripts/build_transportation.sh
```

**Sample script (`build_transportation.sh`)**:
```bash
#!/bin/bash
# Step 1: Sample graphs
python generation/sampling.py --input raw/PeMS.edgelist --output processed/transportation --scale 100

# Step 2: Generate tasks
for task in shortest_path traffic_flow; do
  python generation/task_generator.py \
    --graph_data processed/transportation/100_nodes \
    --template prompts/${task}.txt \
    --output tasks/transportation/${task}.json
done

# Step 3: Validate
python generation/validate.py --tasks tasks/transportation/
```


<span id='Evaluating GraphCogent'/>

## 3. Evaluating GraphCogent <a href='#all_catelogue'>[Back to Top]</a>


**Notice:** Due to the fact that we use 8*4090 GPUs running in parallel during the inference process, our execution script is not suitable for most users. Therefore, we recommend assigning specific tasks to specific GPUs and running each specific task separately.


### **GraphCogent: Unified Execution Framework**  
**Repository Structure**:  
```
GraphCogent/
├── run.py               # Unified execution script
├── eval.py              # Unified evaluation script
├── configs/             # Configuration files
│   ├── graph4real.yaml  # Dataset/task parameters
│   └── llama3.1-8b.yaml # Model settings
└── outputs/             # Results directory
```

---
<span id='Task Execution'/>

#### **3.1 Task Execution**  <a href='#all_catelogue'>[Back to Top]</a>
Run any task (Sensory → Buffer → Execution) with a single command:  
```bash
python run.py \
    --model_path ./models/llama3.1-8b \
    --dataset_config ./configs/graph4real.yaml \
    --task_name shortest_path \          # Specify task (e.g., "node_classification", "cycle_detection")
    --output_dir ./outputs \
    --max_tokens 4096
    --temperature 0.7
    --top_p 1
```

---
<span id='Task Evaluation'/>

#### **3.2 Task Evaluation**  <a href='#all_catelogue'>[Back to Top]</a>
Evaluate results against ground truth:  
```bash
python eval.py \
    --result_dir ./outputs \             # Path to GraphCogent's outputs
    --ground_truth ./data \  # Standard answer JSON path
    --report_file ./outputs/summary.json
```

**Output Format (`summary.json`)**:  
```json
{
  "accuracy": 98.5,
  "tasks": {
    "shortest_path": {"type": "transportation","correct": 98, "total": 100}
  }
}
```

---


<span id='GraphCogent Training Process'/>

## 4. GraphCogent Training Process <a href='#all_catelogue'>[Back to Top]</a>



### Training Pipeline 
Reasoning Agent tuning paradigm consists of four stages: (1) LLaMA-Factory Installation; (2) Thinking Path Collection; (3) LoRA Tuning; (4) DPO Tuning; (5) Export Model.

---

<span id='LLaMA-Factory Installation'/>

#### 4.1 LLaMA-Factory Installation <a href='#all_catelogue'>[Back to Top]</a>

```shell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

**Prerequisites**:  
- Download **Llama3.1-8B-Instruct** weights: [HuggingFace Hub](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)  
- Place under `./models/llama3.1-8b-instruct/`  

---

<span id='Thinking Path Collection'/>

#### 4.2 Thinking Path Collection <a href='#all_catelogue'>[Back to Top]</a>

Generate reasoning trajectories for SFT using Graph4real tasks:
```python
from graphcogent import ThinkingPathGenerator

# Initialize with Graph4real dataset
generator = ThinkingPathGenerator(dataset="Graph4real/train.json")

# Generate <Problem, Reasoning Chain, Tool/Model Decision> triples
generator.run(output_file="data/thinking_paths.jsonl")
```

* **Output Format** (Alpaca-style):
```json
{
  "instruction": "Find shortest path between A and B in [adjacency list]",
  "input": "[[0,1],[1,2],...]",
  "output": "<THINK>Use shortest path tool</THINK><DECISION>tool:shortest_path</DECISION>",
  "system": "You are a graph reasoning expert..."
}
```


* **Prepare data:** Please download our instruction tuning data first. Our fine-tuning dataset follows the Alpaca format. To use LLaMA-Factory for model fine-tuning, we need to add the following three formats to the LLaMA-Factory/data/dataset_info.json path:


```shell
  "Reasoning": {
    "file_name": "${Graph4real/Train/Thinking.json}",
    "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system"
    }
  }
```


---

<span id='LoRA Tuning'/>

#### 4.3 LoRA Tuning <a href='#all_catelogue'>[Back to Top]</a>

Use the following command to run LoRA **fine-tuning** of the Llama3.1-8B-Instruct model under the path: examples/train_lora/llama3_lora_sft.yaml. 

```shell
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```


```shell
# Our fine-tuning parameter settings are as follows.
model_path= Llama3.1/model
output_model= Llama3.1/lora_weight

### model
model_name_or_path: ${model_path}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: Reasoning
template: llama3
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ${output_model}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500

```

---

<span id='DPO Tuning'/>

#### 4.4 DPO Tuning <a href='#all_catelogue'>[Back to Top]</a>
#### Prepare Preference Data:
```python
from graphcogent import DPOPreferenceGenerator

# Generate <chosen, rejected> pairs from thinking paths
DPOPreferenceGenerator(
    sft_output="checkpoints/lora_sft/predictions.jsonl",
    output_file="data/dpo_pairs.jsonl"
).run()
```

#### Config: `examples/train_dpo/graphcogent_dpo.yaml`
```shell
model_path= Llama3.1/model
output_model= saves/llama3.1-8b/lora/dpo+grpo

### model
model_name_or_path: ${model_path}
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: dpo_en_demo
template: llama3
cutoff_len: 4096
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 8

### output
output_dir: {output_model}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 5.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500

```

**Run DPO**:
```shell
llamafactory-cli train examples/train_dpo/graphcogent_dpo.yaml \
  --output_dir ./checkpoints/lora_dpo
```

---

<span id='Export Model'/>

#### 4.5 Export Model <a href='#all_catelogue'>[Back to Top]</a>
Merge LoRA adapters into deployable model:
```shell
llamafactory-cli export \
  --model_name_or_path ./models/llama3-8b-instruct \
  --adapter_path ./checkpoints/lora_dpo \
  --template llama3 \
  --export_dir ./deploy/graphcogent_agent \
  --export_legacy_format false
```

**Verification**:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("./deploy/graphcogent_agent")
assert "graph" in model.config.architectures[0].lower()
```

---

## Folder Structure
```
GraphCogent/
├── data/
│   ├── thinking_paths.jsonl      # SFT data
│   └── dpo_pairs.jsonl           # DPO preferences
├── checkpoints/
│   ├── lora_sft/                 # LoRA weights
│   └── lora_dpo/                 # DPO-optimized
└── deploy/
    └── graphcogent_agent         # Production model
```


