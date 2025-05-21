import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
import transformers
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
model_id = "/home/data2t1/tempuser/Llama-3.1-8B-Instruct"

# Initialize text generation pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Modified to create unweighted graph
def draw_graph(edges, descriptions=None):
    G = nx.Graph()
    for edge in edges:
        if len(edge) >= 2:  # Only need node1 and node2 for unweighted graph
            node1, node2 = edge[0], edge[1]
            G.add_edge(node1, node2)
        else:
            raise ValueError("Each edge must be at least a two-element tuple (node1, node2)")
    return G

# Modified to parse unweighted graphs
def parse_graph(input_data):
    # Regex pattern for unweighted edges: (node1, node2)
    pattern = r"\(\s*('(\d+)'|(\d+))\s*,\s*('(\d+)'|(\d+))\s*\)"
    edges = re.findall(pattern, input_data)

    G = nx.Graph()

    for edge in edges:
        # Extract nodes (handling both quoted and unquoted numbers)
        node1 = edge[0].strip("'") if edge[0] else edge[1]
        node2 = edge[3].strip("'") if edge[3] else edge[4]
        
        # Convert to integers and add edge
        G.add_edge(int(node1), int(node2))

    return G

prompt_file = '/home/data2t1/tempuser/GTAgent/Web/Graph_Agent/GPT_Graph_Agent/prompt5.txt'
with open(prompt_file, 'r', encoding='utf-8') as file:
    prompt = file.read()

output_file_path = '/home/data2t1/tempuser/GTAgent/.Graph4Real/Trans/Question_Json/Small/Web/Shortest_Path.json'
subgraphs_info = read_json_file(output_file_path)

def edge_edit_distance(G, H):
    edges_G = set(G.edges())
    edges_H = set(H.edges())

    edges_to_add = edges_H - edges_G
    edges_to_remove = edges_G - edges_H

    edit_distance = len(edges_to_add) + len(edges_to_remove)

    return edit_distance

def save_intermediate_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=None)

subgraph_results = []
total_retry = 0
total_ed = 0
num = 0
output_json_path = 'zNack_test/Graph_representaion/Result/50_edges_Web.json'

for index, subgraph_info in tqdm(enumerate(subgraphs_info), total=len(subgraphs_info), desc="Processing Subgraphs"):
    print(f"Processing Subgraph {index + 1}")
    cnt = 0
    retry = 0
    num += 1
    
    current_subgraph_result = {
        "index": index,
        "description": subgraph_info['Description'],
        "pairs": []
    }
    
    for edges, descriptions in zip(subgraph_info['Edges'], subgraph_info['Description']):
        description = "G: " + ", ".join(descriptions) + "."
        messages = []
        messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": description})    
        
        G = None    
        H = draw_graph(edges, descriptions)   
        
    
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=8192,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=1,
        )
        first = outputs[0]["generated_text"][-1]['content']
        print(first)
        G = parse_graph(first)
        
        print(f'Graph G edges: {G.edges()}')
        print(f'Graph H edges: {H.edges()}')
        
        ed = edge_edit_distance(G, H)
        print(ed)
        total_ed += ed
        total_retry += retry
        
        current_pair = {
            "G_edges": list(G.edges()),
            "H_edges": list(H.edges()),
            "edit_distance": ed,
            "retry": retry,
        }
        current_subgraph_result["pairs"].append(current_pair)
        
        save_intermediate_results(subgraph_results + [current_subgraph_result], output_json_path)
        
        print(f'total cnt {cnt}, total_ed {total_ed}, total_retry {retry}')
    
    current_subgraph_result["total_edit_distance"] = total_ed
    current_subgraph_result["total_retry_times"] = total_retry
    subgraph_results.append(current_subgraph_result)
    
    save_intermediate_results(subgraph_results, output_json_path)
    
    print(f'total_ed is {total_ed}')

print("Final results saved to:", output_json_path)
