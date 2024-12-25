import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict
from ..model.transformer import ShiemiTransformer
from ..tokenizer.tokenizer import ShiemiTokenizer


class TextGenerator:
    def __init__(
        self,
        model: ShiemiTransformer,
        tokenizer: ShiemiTokenizer,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> List[str]:
        """Generate text based on a prompt."""
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        # Set special token IDs
        pad_token_id = pad_token_id or self.tokenizer.pad_token_id
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id

        # Track generated sequences
        generated_sequences = []

        for _ in range(num_return_sequences):
            curr_input_ids = input_ids.clone()

            for _ in range(max_length):
                # Get model outputs
                outputs = self.model(curr_input_ids)
                next_token_logits = outputs[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for seq in curr_input_ids:
                        for token_id in set(seq.tolist()):
                            next_token_logits[:,
                                              token_id] /= repetition_penalty

                # Filter with top-k
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Filter with top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[...,
                                             1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(
                        next_token_logits, dim=-1, keepdim=True)

                # Append next token
                curr_input_ids = torch.cat(
                    [curr_input_ids, next_token], dim=-1)

                # Check if we've hit the EOS token
                if next_token[0, 0].item() == eos_token_id:
                    break

            # Decode the generated sequence
            generated_sequence = curr_input_ids[0].tolist()
            text = self.tokenizer.decode(
                generated_sequence, skip_special_tokens=True)
            generated_sequences.append(text)

        return generated_sequences

    def chat(
        self,
        message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **generation_kwargs
    ) -> str:
        """Have a conversation with the model."""
        if chat_history is None:
            chat_history = []

        # Format the conversation history and current message
        prompt = ""
        for entry in chat_history[-5:]:  # Keep last 5 messages for context
            if "user" in entry:
                prompt += f"User: {entry['user']}\n"
            if "assistant" in entry:
                prompt += f"Shiemi: {entry['assistant']}\n"

        prompt += f"User: {message}\nShiemi:"

        # Generate response
        response = self.generate(prompt, **generation_kwargs)[0]

        # Clean up the response
        if "User:" in response:
            response = response.split("User:")[0].strip()

        return response.strip()

    def __call__(self, *args, **kwargs):
        """Convenience method to call generate."""
        return self.generate(*args, **kwargs)
