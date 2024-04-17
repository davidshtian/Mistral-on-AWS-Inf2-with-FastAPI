import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers_neuronx import MistralForSampling, GQA, NeuronConfig

# Set sharding strategy for GQA to be shard over heads
neuron_config = NeuronConfig(
    group_query_attention=GQA.SHARD_OVER_HEADS
)

# Create and compile the Neuron model
model_neuron = MistralForSampling.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', amp='bf16', batch_size=1, tp_degree=2, n_positions=2048, neuron_config=neuron_config)
model_neuron.to_neuron()

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
tokenizer.pad_token_id = tokenizer.eos_token_id

input_prompt = 'Who are you?'

input_prompt = "[INST] " + input_prompt + " [/INST]"
encoded_input = tokenizer(input_prompt, return_tensors='pt')
original_input_ids = encoded_input.input_ids
input_ids_length = original_input_ids.shape[1]
power_of_length = 64
while power_of_length < input_ids_length:
    power_of_length *= 2
padding_size = ((input_ids_length - 1) // 64 + 1) * power_of_length
padding_gap = padding_size - input_ids_length
padded_input_ids = F.pad(original_input_ids, (padding_gap, 0), value=tokenizer.pad_token_id)

input_embeds = model_neuron.chkpt_model.model.embed_tokens(padded_input_ids)

input_embeds_np = input_embeds.detach().numpy()
np.save('./input_embeds.npy', input_embeds_np)
