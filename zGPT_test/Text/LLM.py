# import json
# from openai import OpenAI
# from tqdm import tqdm
# import os

# GPT_MODEL = "gpt-3.5-turbo-0125"


# client = OpenAI(
#    api_key="sk-7m964UeoHpTlYtwe568a6e356fB749359cE57a7569344eF3",
#    base_url="https://api.gptapi.us/v1/chat/completions"
# )

# def chat_completion_request(messages, model=GPT_MODEL):
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=messages,
#             max_tokens=8192,
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e


# final_answer_file_path = '/home/data2t2/wrz/Graph Tools/1closedLLM-COT/All_type_mission/once/cycle_check_graphExistance/apicall_gpt3.5.json'

# data = []
# if os.path.exists(final_answer_file_path):
#     with open(final_answer_file_path, 'r') as file:
#         try:
#             data = json.load(file)
#         except json.JSONDecodeError:
#             data = []

# with open('/home/data2t2/wrz/Graph Tools/1closedLLM-COT/All_type_mission/once/cycle_check_graphExistance/cycle_simple.json', 'r', encoding='utf-8') as file:
#     files = json.load(file)

# # # Read the custom system prompt from CoT-prompt.txt
# # cot_prompt_path = '/home/data2t2/wrz/Graph Tools/1closedLLM-COT/All_type_mission/once/cycle_check_graphExistance/CoT-prompt.txt'
# # with open(cot_prompt_path, 'r', encoding='utf-8') as file:
# #     cot_prompt = file.read()

# cnt = 0

# cot_prompt = "In the context of a graph reasoning problem for a transportation network, the format [15347, 235, 22] represents node 15347 connected to node 235 with an edge weight of 22. Please answer the following question based on the given graph and problem."

# for obj in tqdm(files):
#     cnt += 1
#     message = obj['prompt']
#     answer = obj['answer']
#     messages = []
#     messages.append({"role": "system", "content": cot_prompt})
#     messages.append({"role": "user", "content": message})
#     chat_response = chat_completion_request(messages)
#     print(chat_response)
    
#     if isinstance(chat_response, Exception):
#         # Skip the current iteration if an error occurred
#         continue
    
#     # Extracting the actual response content using dot notation
#     first = chat_response.choices[0].message.content


#     messages.append({"role": "assistant", "content": first})
#     messages.append({"role": "user", "content": "Only give the answer True or False, no any other words"})

#     chat_response = chat_completion_request(messages)
#     print(chat_response)
    
#     if isinstance(chat_response, Exception):
#         # Skip the current iteration if an error occurred
#         continue
    
#     # Extracting the actual response content using dot notation
#     second = chat_response.choices[0].message.content

#     data.append({
#         'prompt': message,
#         'first': first,
#         'second': second,
#         'answer': answer,
#     })

#     with open(final_answer_file_path, 'w') as file:
#         json.dump(data, file, indent=4)

# print("JSON 文件已更新。")

import json
from openai import OpenAI
from tqdm import tqdm
import os

GPT_MODEL = "gpt-4o-2024-08-06"

client = OpenAI(
   api_key="sk-Hv9ensWblsslUzoI13C0A69bD2094fB0B6CeDb81709e17D5",
   base_url="https://api.holdai.top/v1"
)

def chat_completion_request(messages, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=8192,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# Specify output directory
output_dir = 'zGPT_test/Text/Trans_small'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

input_file_path = '.Graph4Real/Trans/Json/Small/Trans/Edge_Count.json'

# Get the input file name without extension
input_file_name = os.path.splitext(os.path.basename(input_file_path))[0]
final_answer_file_path = os.path.join(output_dir, f"{input_file_name}_results.json")

data = []
if os.path.exists(final_answer_file_path):
    with open(final_answer_file_path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []

with open(input_file_path, 'r', encoding='utf-8') as file:
    files = json.load(file)

cot_prompt = "In the context of a graph reasoning problem for a transportation network, the format [15347, 235, 22] represents node 15347 connected to node 235 with an edge weight of 22. Please answer the following question based on the given graph and problem."

cnt = 0

for obj in tqdm(files):
    cnt += 1
    message = obj['question']
    answer = obj['answer']
    messages = []
    messages.append({"role": "system", "content": cot_prompt})
    messages.append({"role": "user", "content": message})
    chat_response = chat_completion_request(messages)
    print(chat_response)
    
    if isinstance(chat_response, Exception):
        continue
    
    first = chat_response.choices[0].message.content

    messages.append({"role": "assistant", "content": first})
    messages.append({"role": "user", "content": "Only give the answer True or False, no any other words"})

    chat_response = chat_completion_request(messages)
    print(chat_response)
    
    if isinstance(chat_response, Exception):
        continue
    
    second = chat_response.choices[0].message.content

    data.append({
        'prompt': message,
        'first': first,
        'second': second,
        'answer': answer,
    })

    with open(final_answer_file_path, 'w') as file:
        json.dump(data, file, indent=4)

print(f"JSON 文件已保存到: {final_answer_file_path}")
