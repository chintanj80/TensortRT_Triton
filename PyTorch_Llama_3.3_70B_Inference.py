import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Llama33Inference:
    def __init__(self, device="cuda"):
        """
        Initialize the Llama 3.3 70B model for inference.
        
        Args:
            device: The device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Set model ID for Llama 3.3 70B
        self.model_id = "meta-llama/Meta-Llama-3.3-70B"
        
        # Load tokenizer and model
        logger.info(f"Loading tokenizer from {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        logger.info(f"Loading model from {self.model_id}")
        # If running on lower VRAM GPU, enable 4-bit quantization with bitsandbytes
        if self.device == "cuda":
            try:
                import bitsandbytes as bnb
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config={"load_in_4bit": True}
                )
                logger.info("Model loaded with 4-bit quantization")
            except ImportError:
                logger.info("bitsandbytes not available, loading model in bfloat16")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map={"": self.device}
            )
        
        logger.info("Model initialized successfully")
    
    def generate_response(
        self, 
        question: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response for the given question.
        
        Args:
            question: The user's question.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness. Lower = more deterministic.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            system_prompt: Optional system prompt for instruction.
            
        Returns:
            The generated response.
        """
        # Format the input with appropriate Llama 3.3 chat template
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        else:
            messages = [{"role": "user", "content": question}]
            
        # Apply the chat template to format the messages
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return response
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<assistant>" in generated_text:
            response = generated_text.split("<assistant>")[-1].strip()
        else:
            # Fallback to returning full text minus input
            response = generated_text[len(input_text):].strip()
            
        return response

    def batch_generate(
        self, 
        questions: List[str], 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of questions.
        
        Args:
            questions: List of user questions.
            system_prompt: Optional system prompt to use for all questions.
            **kwargs: Additional parameters to pass to generate_response.
            
        Returns:
            List of generated responses.
        """
        return [self.generate_response(q, system_prompt=system_prompt, **kwargs) for q in questions]


# Example usage
if __name__ == "__main__":
    # Initialize the model
    llm = Llama33Inference(device="cuda")  # Use "cpu" if no GPU available
    
    # Set a system prompt (optional)
    system_prompt = """You are a helpful, harmless, and honest AI assistant. 
    Always provide accurate information and admit when you don't know something."""
    
    # Ask a question
    question = "What are the main differences between transformer-based and RNN-based language models?"
    
    print(f"Question: {question}")
    response = llm.generate_response(
        question,
        system_prompt=system_prompt,
        max_new_tokens=256,
        temperature=0.7
    )
    print(f"Response: {response}")
    
    # Batch processing example
    questions = [
        "Explain quantum computing in simple terms.",
        "What's the capital of France and some interesting facts about it?",
        "How do neural networks learn?"
    ]
    
    responses = llm.batch_generate(
        questions,
        system_prompt=system_prompt,
        max_new_tokens=128
    )
    
    for q, r in zip(questions, responses):
        print(f"\nQuestion: {q}")
        print(f"Response: {r}")
