import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Dict, Any
import logging
import time
import statistics
import json
from datetime import datetime

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
        system_prompt: Optional[str] = None,
        measure_time: bool = False,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response for the given question with metrics.
        
        Args:
            question: The user's question.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness. Lower = more deterministic.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            system_prompt: Optional system prompt for instruction.
            measure_time: Whether to measure and return timing metrics.
            stream: Whether to use streaming generation to measure time to first token.
            
        Returns:
            Dictionary with generated response and metrics (if measure_time=True).
        """
        metrics = {"input_tokens": 0, "output_tokens": 0} if measure_time else {}
        start_time = time.time() if measure_time else None
        
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
        
        if measure_time:
            metrics["input_tokens"] = len(inputs.input_ids[0])
            metrics["tokenization_time"] = time.time() - start_time
            metrics["preprocessing_time"] = metrics["tokenization_time"]
        
        # Generate response
        gen_start_time = time.time() if measure_time else None
        
        try:
            with torch.no_grad():
                if stream and measure_time:
                    # Stream to measure time to first token
                    try:
                        from transformers import TextIteratorStreamer
                        from threading import Thread
                        
                        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
                        ttft_streamer = TimeToFirstTokenStreamer(self.tokenizer)
                        
                        # Set up generation arguments
                        generation_kwargs = dict(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=temperature > 0,
                            pad_token_id=self.tokenizer.eos_token_id,
                            streamer=streamer
                        )
                        
                        # Create a thread for generation
                        generation_thread = Thread(
                            target=self.model.generate, 
                            kwargs=generation_kwargs
                        )
                        generation_thread.start()
                        
                        # Use ttft_streamer as a side-channel to measure first token
                        tokens_so_far = 0
                        first_token_time = None
                        for token in streamer:
                            tokens_so_far += 1
                            if tokens_so_far == 1:
                                first_token_time = time.time() - gen_start_time
                                ttft_streamer.put(token)
                            
                        generation_thread.join()
                        output = inputs.input_ids  # Placeholder, we don't need the actual output
                        
                        if measure_time and first_token_time is not None:
                            metrics["time_to_first_token"] = first_token_time
                        
                    except ImportError:
                        logger.warning("TextIteratorStreamer not available, falling back to regular generation")
                        output = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            do_sample=temperature > 0,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                else:
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA out of memory. Try reducing batch size or model precision.")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            raise e
        
        if measure_time:
            metrics["generation_time"] = time.time() - gen_start_time
            metrics["output_tokens"] = len(output[0]) - len(inputs.input_ids[0])
            metrics["tokens_per_second"] = metrics["output_tokens"] / metrics["generation_time"]
        
        # Decode response
        decode_start = time.time() if measure_time else None
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<assistant>" in generated_text:
            response = generated_text.split("<assistant>")[-1].strip()
        else:
            # Fallback to returning full text minus input
            response = generated_text[len(input_text):].strip()
        
        if measure_time:
            metrics["decoding_time"] = time.time() - decode_start
            metrics["total_time"] = time.time() - start_time
        
        result = {"response": response}
        if measure_time:
            result["metrics"] = metrics
            
        return result

    def batch_generate(
        self, 
        questions: List[str], 
        system_prompt: Optional[str] = None,
        measure_time: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of questions.
        
        Args:
            questions: List of user questions.
            system_prompt: Optional system prompt to use for all questions.
            measure_time: Whether to measure and return timing metrics.
            **kwargs: Additional parameters to pass to generate_response.
            
        Returns:
            List of generated responses with metrics (if measure_time=True).
        """
        return [
            self.generate_response(
                q, 
                system_prompt=system_prompt, 
                measure_time=measure_time, 
                **kwargs
            ) 
            for q in questions
        ]

    def run_benchmark(
        self,
        question: str,
        num_runs: int = 100,
        system_prompt: Optional[str] = None,
        warmup_runs: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run benchmark tests for inference performance.
        
        Args:
            question: The question to use for benchmarking.
            num_runs: Number of inference runs to perform.
            system_prompt: Optional system prompt.
            warmup_runs: Number of warmup runs to perform (not counted in statistics).
            **kwargs: Additional parameters to pass to generate_response.
            
        Returns:
            Dictionary with benchmark results and statistics.
        """
        logger.info(f"Starting benchmark with {num_runs} runs (plus {warmup_runs} warmup runs)")
        
        # Perform warmup runs to stabilize GPU performance
        if warmup_runs > 0:
            logger.info(f"Performing {warmup_runs} warmup runs...")
            for i in range(warmup_runs):
                try:
                    _ = self.generate_response(
                        question, 
                        system_prompt=system_prompt,
                        measure_time=False,
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"Error during warmup run {i+1}: {str(e)}")
                    if "CUDA out of memory" in str(e) and self.device == "cuda":
                        torch.cuda.empty_cache()
        
        # Perform actual benchmark runs
        runs = []
        success_count = 0
        
        for i in range(num_runs):
            try:
                logger.info(f"Run {i+1}/{num_runs}")
                result = self.generate_response(
                    question, 
                    system_prompt=system_prompt, 
                    measure_time=True,
                    **kwargs
                )
                runs.append(result["metrics"])
                success_count += 1
            except Exception as e:
                logger.error(f"Error during benchmark run {i+1}: {str(e)}")
                if "CUDA out of memory" in str(e) and self.device == "cuda":
                    torch.cuda.empty_cache()
        
        if not runs:
            logger.error("No successful benchmark runs were completed")
            return {"error": "No successful benchmark runs", "success_count": 0}
        
        # Calculate statistics
        stats = {}
        metric_keys = runs[0].keys()
        
        for key in metric_keys:
            values = [run[key] for run in runs if key in run]
            if not values:
                continue
                
            stats[key] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0
            }
        
        # Add device and timestamp information
        benchmark_results = {
            "device": self.device,
            "device_info": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model": self.model_id,
            "num_runs": num_runs,
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "system_prompt": system_prompt,
            "parameters": kwargs,
            "statistics": stats,
            "runs": runs
        }
        
        return benchmark_results


# Custom streamer for measuring time to first token
class TimeToFirstTokenStreamer:
    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.first_token_time = None
        self.tokens_generated = 0
        self.start_time = time.time()
        
    def put(self, text_idx):
        # Called every time a new token is generated
        self.tokens_generated += 1
        if self.tokens_generated == 1 and self.first_token_time is None:
            self.first_token_time = time.time() - self.start_time
    
    def end(self):
        # Called at the end of generation
        pass


# Example usage
if __name__ == "__main__":
    # Initialize the model
    llm = Llama33Inference(device="cuda")  # Use "cpu" if no GPU available
    
    # Set a system prompt (optional)
    system_prompt = """You are a helpful, harmless, and honest AI assistant. 
    Always provide accurate information and admit when you don't know something."""
    
    # Define a test question for the benchmark
    test_question = "What are the main differences between transformer-based and RNN-based language models?"
    
    # Set generation parameters
    generation_params = {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    }
    
    # Run single test with timing to make sure everything works
    print("Running test inference...")
    test_result = llm.generate_response(
        test_question,
        system_prompt=system_prompt,
        measure_time=True,
        **generation_params
    )
    
    print(f"Test response: {test_result['response'][:100]}...")
    print(f"Test metrics: {json.dumps(test_result['metrics'], indent=2)}")
    
    # Run the benchmark (100 times)
    print("\nRunning full benchmark...")
    benchmark_results = llm.run_benchmark(
        test_question,
        num_runs=100,
        system_prompt=system_prompt,
        **generation_params
    )
    
    # Save benchmark results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"llama33_benchmark_{timestamp}.json"
    
    with open(result_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Benchmark results saved to {result_file}")
    
    # Print summary statistics
    print("\nSummary statistics:")
    for metric, stats in benchmark_results["statistics"].items():
        print(f"{metric}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Min: {stats['min']:.4f}")
        print(f"  Max: {stats['max']:.4f}")
        print(f"  StdDev: {stats['stdev']:.4f}")
        print()
    
    # Batch processing example
    print("\nRunning batch inference example...")
    questions = [
        "Explain quantum computing in simple terms.",
        "What's the capital of France and some interesting facts about it?",
        "How do neural networks learn?"
    ]
    
    batch_results = llm.batch_generate(
        questions,
        system_prompt=system_prompt,
        measure_time=True,
        **generation_params
    )
    
    for i, (q, r) in enumerate(zip(questions, batch_results)):
        print(f"\nQuestion {i+1}: {q}")
        print(f"Response: {r['response'][:100]}...")
        print(f"Total time: {r['metrics']['total_time']:.4f} seconds")
