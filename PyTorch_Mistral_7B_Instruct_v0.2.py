# Mistral 7B Instruct v0.2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import statistics
import psutil
import gc
import numpy as np
import json
from datetime import datetime
import os

class MistralInstructBot:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the Mistral 7B Instruct model and tokenizer.
        
        Args:
            model_id: The model identifier on Hugging Face Hub.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print(f"Loading model and tokenizer for {model_id}...")
        
        # Record memory before loading
        self.initial_memory = self._get_memory_usage()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with appropriate parameters for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Record memory after loading
        self.post_load_memory = self._get_memory_usage()
        self.model_memory_footprint = self.post_load_memory['gpu'] - self.initial_memory['gpu']
        
        print(f"Model loaded. GPU memory used by model: {self.model_memory_footprint:.2f} MB")
    
    def _get_memory_usage(self):
        """Get current memory usage for CPU and GPU."""
        memory_stats = {
            'cpu': psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        }
        
        if torch.cuda.is_available():
            memory_stats['gpu'] = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        else:
            memory_stats['gpu'] = 0
            
        return memory_stats
    
    def format_prompt(self, question, instruction=None):
        """Format the input following Mistral's expected chat template."""
        if instruction:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question}
            ]
        else:
            messages = [
                {"role": "user", "content": question}
            ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def answer_question(self, question, instruction=None, max_new_tokens=512, temperature=0.7):
        """Get an answer from the Mistral model for a given question."""
        # Format the prompt
        prompt = self.format_prompt(question, instruction)
        
        # Record stats before generation
        memory_before = self._get_memory_usage()
        tokens_before = len(self.tokenizer.encode(prompt))
        start_time = time.time()
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response, skipping the input prompt
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Record stats after generation
        end_time = time.time()
        memory_after = self._get_memory_usage()
        tokens_generated = len(self.tokenizer.encode(response))
        
        # Calculate metrics
        stats = {
            'time_seconds': end_time - start_time,
            'tokens_input': tokens_before,
            'tokens_output': tokens_generated,
            'tokens_per_second': tokens_generated / (end_time - start_time),
            'memory_increase_mb': {
                'cpu': memory_after['cpu'] - memory_before['cpu'],
                'gpu': memory_after['gpu'] - memory_before['gpu']
            }
        }
        
        return response, stats

    def run_performance_test(self, questions, n_runs=100, instruction=None):
        """
        Run performance tests by answering each question multiple times.
        
        Args:
            questions: List of questions to test
            n_runs: Number of times to run each question
            instruction: Optional system instruction
            
        Returns:
            Dictionary with performance statistics
        """
        results = {}
        
        for question in questions:
            print(f"\nTesting question: '{question}'")
            question_stats = {
                'responses': [],
                'times': [],
                'tokens_per_second': [],
                'memory_increases': {'cpu': [], 'gpu': []},
                'token_counts': []
            }
            
            for i in range(n_runs):
                if i % 10 == 0:
                    print(f"  Run {i+1}/{n_runs}...")
                
                # Clear CUDA cache between runs for more consistent measurements
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                response, stats = self.answer_question(question, instruction)
                
                # Store results
                question_stats['responses'].append(response)
                question_stats['times'].append(stats['time_seconds'])
                question_stats['tokens_per_second'].append(stats['tokens_per_second'])
                question_stats['memory_increases']['cpu'].append(stats['memory_increase_mb']['cpu'])
                question_stats['memory_increases']['gpu'].append(stats['memory_increase_mb']['gpu'])
                question_stats['token_counts'].append(stats['tokens_output'])
            
            # Calculate aggregate statistics
            results[question] = {
                'response_sample': question_stats['responses'][0],  # Just the first response as sample
                'response_uniqueness': len(set(question_stats['responses'])) / len(question_stats['responses']),
                'time_stats': {
                    'mean': statistics.mean(question_stats['times']),
                    'median': statistics.median(question_stats['times']),
                    'min': min(question_stats['times']),
                    'max': max(question_stats['times']),
                    'std_dev': statistics.stdev(question_stats['times']) if len(question_stats['times']) > 1 else 0
                },
                'tokens_per_second_stats': {
                    'mean': statistics.mean(question_stats['tokens_per_second']),
                    'median': statistics.median(question_stats['tokens_per_second']),
                    'min': min(question_stats['tokens_per_second']),
                    'max': max(question_stats['tokens_per_second'])
                },
                'memory_usage_mb': {
                    'cpu': {
                        'mean': statistics.mean(question_stats['memory_increases']['cpu']),
                        'max': max(question_stats['memory_increases']['cpu'])
                    },
                    'gpu': {
                        'mean': statistics.mean(question_stats['memory_increases']['gpu']),
                        'max': max(question_stats['memory_increases']['gpu'])
                    }
                },
                'token_count_stats': {
                    'mean': statistics.mean(question_stats['token_counts']),
                    'min': min(question_stats['token_counts']),
                    'max': max(question_stats['token_counts']),
                    'std_dev': statistics.stdev(question_stats['token_counts']) if len(question_stats['token_counts']) > 1 else 0
                }
            }
            
            print(f"  Completed testing for question. Avg time: {results[question]['time_stats']['mean']:.2f}s")
        
        return results

    def save_performance_results(self, results, filename=None):
        """Save performance results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mistral_performance_{timestamp}.json"
        
        # Add system info to results
        system_info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'model_memory_footprint_mb': self.model_memory_footprint
        }
        
        full_results = {
            'system_info': system_info,
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nPerformance results saved to {filename}")
        return filename


def main():
    # Create an instance of MistralInstructBot
    mistral_bot = MistralInstructBot()
    
    # Define test questions of varying complexity
    test_questions = [
        "What is the capital of France?",  # Simple, factual
        "Explain the concept of neural networks in simple terms.",  # Moderate complexity
        "Write a short story about a robot discovering emotions.",  # Creative, longer output
        "Summarize the key events of World War II.",  # Complex, knowledge-intensive
    ]
    
    # Run performance test (adjust n_runs as needed)
    n_runs = 100  # Set to 100 for your full test
    results = mistral_bot.run_performance_test(test_questions, n_runs=n_runs)
    
    # Save results
    results_file = mistral_bot.save_performance_results(results)
    
    # Print summary of results
    print("\n===== PERFORMANCE SUMMARY =====")
    for question, stats in results.items():
        print(f"\nQuestion: {question}")
        print(f"  Avg. Time: {stats['time_stats']['mean']:.2f}s (±{stats['time_stats']['std_dev']:.2f}s)")
        print(f"  Avg. Tokens/s: {stats['tokens_per_second_stats']['mean']:.2f}")
        print(f"  Avg. Output Length: {stats['token_count_stats']['mean']:.1f} tokens")
        print(f"  Response Uniqueness: {stats['response_uniqueness']*100:.1f}%")
    
    # Calculate overall statistics
    avg_time = statistics.mean([stats['time_stats']['mean'] for stats in results.values()])
    avg_tokens_per_second = statistics.mean([stats['tokens_per_second_stats']['mean'] for stats in results.values()])
    
    print("\n===== OVERALL STATISTICS =====")
    print(f"Average Response Time: {avg_time:.2f}s")
    print(f"Average Generation Speed: {avg_tokens_per_second:.2f} tokens/second")
    print(f"Full results saved to: {results_file}")


if __name__ == "__main__":
    main()