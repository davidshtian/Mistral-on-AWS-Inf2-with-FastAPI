import logging
from typing import Union, List, Optional, Dict, Any, Literal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers_neuronx import MistralForSampling, GQA, NeuronConfig

class MistralModel:
    """
    A class for generating text using the Mistral language model.
    """

    def __init__(self):
        self.neuron_config = NeuronConfig(group_query_attention=GQA.SHARD_OVER_HEADS)
        self.model_name = 'mistralai/Mistral-7B-Instruct-v0.2'
        self.amp: Literal['bf16', 'fp32'] = 'bf16'
        self.batch_size = 1
        self.tp_degree = 2
        self.n_positions = 2048

        self.model = self._load_model()
        self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.prompt_template = "<s>[INST] {prompt} [/INST]"

    def _load_model(self) -> MistralForSampling:
        """
        Load and initialize the Mistral model.

        Returns:
            MistralForSampling: The initialized Mistral model.
        """
        model = MistralForSampling.from_pretrained(
            self.model_name,
            amp=self.amp,
            batch_size=self.batch_size,
            tp_degree=self.tp_degree,
            n_positions=self.n_positions,
            neuron_config=self.neuron_config
        )
        model.to_neuron()
        return model

    def generate(self, inputs: Union[str, List[int]], parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate text using the Mistral model.

        Args:
            inputs (Union[str, List[int]]): The input prompt or a list of input embeddings.
            parameters (Optional[Dict[str, Any]]): Optional parameters for text generation.

        Returns:
            str: The generated text.

        Raises:
            ValueError: If the input type is invalid.
        """
        try:
            max_new_tokens = parameters.get("max_new_tokens", 256)

            if isinstance(inputs, str):
                generated_text = self._generate_from_prompt(inputs, max_new_tokens)
            elif isinstance(inputs, list):
                generated_text = self._generate_from_embeddings(inputs, max_new_tokens)
            else:
                raise ValueError("Invalid input type. Must be str or List[int]")

            return generated_text
        except Exception as e:
            logging.error(f"Error generating text: {e}")
            raise

    def _generate_from_prompt(self, prompt: str, max_new_tokens: int) -> str:
        """
        Generate text from a given prompt using the Mistral model.

        Args:
            prompt (str): The input prompt.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        input_prompt = self.prompt_template.format(prompt=prompt)
        encoded_input = self.tokenizer(input_prompt, return_tensors='pt')
        input_ids = encoded_input.input_ids

        with torch.inference_mode():
            generated_sequence = self.model.sample(input_ids, sequence_length=min(self.n_positions, input_ids.shape[1]+max_new_tokens), start_ids=None)
            decoded_output = [self.tokenizer.decode(tok) for tok in generated_sequence]

        generated_text = decoded_output[0].split('[/INST]')[1].strip("</s>").strip()
        return generated_text

    def _generate_from_embeddings(self, input_embeddings: List[int], max_new_tokens: int) -> str:
        """
        Generate text from a given list of input embeddings using the Mistral model.

        Args:
            input_embeddings (List[int]): A list of input embeddings.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        input_embeds_tensor = torch.tensor(input_embeddings)
        input_embeds_length = input_embeds_tensor.shape[1]
        power_of_length = 64
        padding_size = ((input_embeds_length - 1) // 64 + 1) * power_of_length
        padding_gap = padding_size - input_embeds_length
        padded_input_embeds = F.pad(input_embeds_tensor, (0, 0, padding_gap, 0), value=self.tokenizer.pad_token_id)

        with torch.inference_mode():
            generated_sequence = self.model.sample(padded_input_embeds, sequence_length=min(self.n_positions, padding_size+max_new_tokens), start_ids=None)
            decoded_output = [self.tokenizer.decode(tok) for tok in generated_sequence]

        generated_text = decoded_output[0].strip("</s>").strip()
        return generated_text
