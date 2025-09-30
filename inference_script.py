#!/usr/bin/env python3
"""
Self-contained script for model inference and Lean code compilation.
Loads model from specified path, performs vLLM inference, and processes results.
"""

import os
import json
import time
import uuid
import hashlib
import re
from typing import List, Dict, Any, Optional
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# Constants
REQUEST_DIR = os.path.expanduser("/home/bl3615/data/shared_a/requests")
RESPONSE_DIR = os.path.expanduser("/home/bl3615/data/shared_a/responses")
TIMEOUT = 600


def remove_comments(text: str) -> str:
    """Remove comments from text"""
    # First remove all /- ... -/ blocks
    text = re.sub(r'/-{1,2} (?!special open -/).*?-{1,2}/', '', text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split('--', 1)[0]
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text.strip()


def remove_specific_lines(text: str) -> str:
    """Remove import, set_option, and open lines"""
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not (
        line.strip().startswith('import') or
        line.strip().startswith('set_option') or
        line.strip().startswith('open')
    )]
    return '\n'.join(filtered_lines)


def return_theorem_to_prove(text: str) -> Optional[str]:
    """Extract theorem to prove from text"""
    pattern = r'((?:theorem).*?:=\s*by(?:\s*sorry)?)'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def return_theorem_to_replace(text: str) -> Optional[tuple]:
    """Find theorem to replace in text"""
    pattern = r'((?:^|\s)theorem\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def replace_statement_in_proof(statement: str, proof: str) -> str:
    """Replace statement in proof"""
    statement_str = return_theorem_to_prove(remove_comments(statement))
    if statement_str is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n') if x is not None])
        return f"[[Error]], can not find 'theorem' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof_str.split('\n') if x is not None])
        return f"[[Error]], can not find 'theorem' and ':=' in {error_app}"
    return proof_str[:span[0]] + statement_str.strip("sorry") + proof_str[span[1]:]


class DeepSeekCoTHandler:
    """Handler for DeepSeek CoT format"""
    
    def __init__(self):
        pass
    
    def extrac_code(self, inputs: str) -> str:
        """Extract Lean code from inputs"""
        import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
        
        # Try different patterns to extract Lean code
        patterns = [
            r'```lean4\n(.*?)\n```',
            r'```lean4\n(.*?)```',
            r'```lean\n(.*?)```'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, inputs, re.DOTALL)
            if matches:
                return import_head + matches[-1]
        
        return "None"
    
    def extract_original_lean4_code(self, inputs: str) -> str:
        """Extract original Lean4 code from inputs"""
        # First extract content between "user" and "assistant"
        user_assistant_pattern = r"\buser\b\s*([\s\S]*?)\s*\bassistant\b"
        user_content = re.findall(user_assistant_pattern, inputs, re.DOTALL)
        
        if not user_content:
            print(f"[[Warning]] No content found between user and assistant tags")
            return ""
            
        # Then extract Lean4 code from the first user content
        return_code = self.extrac_code(user_content[0])
        if return_code == "None":
            print(f"[[Warning]] No Lean 4 code found in the input")
            return ""
        else:
            return return_code
    
    def problem_check(self, statement: str, full_code: str) -> str:
        """Check and process the problem"""
        full_code = replace_statement_in_proof(statement, full_code)
        return full_code


def generate_name(code: str, index: int) -> str:
    """Generate a unique name based on the index and hash value of the string"""
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    return f"{index}_{code_hash}"


def compile_lean(value_list: List[str], timeout: int = TIMEOUT) -> Optional[List[Dict]]:
    """Compile Lean code using the compiler service"""
    if not isinstance(value_list, list) or not all(isinstance(v, str) for v in value_list):
        raise ValueError("Input must be a list of strings")

    task_id = str(uuid.uuid4())
    task_data = {
        "task_id": task_id,
        "tasks": [],
        "proof_timeout": 150
    }

    # Generate tasks with unique names
    for i, code in enumerate(value_list):
        task_data["tasks"].append({
            "id": i,
            "name": generate_name(code, i),
            "code": code
        })

    task_file = os.path.join(REQUEST_DIR, f"{task_id}.json")
    result = None

    with open(task_file, "w") as f:
        json.dump(task_data, f)
    print(f"[Node A] Submitted task {task_id}, number of tasks: {len(value_list)}")

    start_time = time.time()
    response_file = os.path.join(RESPONSE_DIR, f"{task_id}.json")

    results = None

    try:
        while time.time() - start_time < timeout:
            if os.path.exists(response_file):
                time.sleep(1)
                with open(response_file, "r") as f:
                    response_data = json.load(f)

                results = response_data.get("results", [])
                # Sort results based on the index in the name field
                results = sorted(results, key=lambda x: int(x["name"].split('_')[0]))
                print(f"[Node A] Task {task_id} completed")
                break
            time.sleep(1)

        if results is None:
            print(f"[Node A] Task {task_id} timed out without returning!")
    except Exception as e:
        print(f"[Node A] Error occurred while processing task {task_id}: {e}")
        results = None
    finally:
        try:
            if os.path.exists(task_file):
                os.remove(task_file)
                print(f"[Node A] Request file {task_file} deleted")
        except Exception as e:
            print(f"[Node A] Failed to delete request file {task_file}: {e}")

    return results


def process_solution_strings(solution_strs: List[str]) -> List[str]:
    """Process solution strings to extract Lean code for compilation"""
    handler = DeepSeekCoTHandler()
    code_to_compile_list = []
    
    for solution_str in solution_strs:
        try:
            # Clean the solution string
            solution_str_clean = remove_comments(solution_str)
            
            # Extract the proof from the solution
            full_proof = handler.extrac_code(solution_str_clean)
            
            # Extract original Lean4 code from input
            original_lean4_code_in_input = handler.extract_original_lean4_code(solution_str_clean)
            
            # Check and process the problem
            full_code = handler.problem_check(original_lean4_code_in_input, full_proof)
            
            if full_code and "[[Error]]" not in full_code:
                # Extract lean4 code block for compilation
                code_to_compile = remove_specific_lines(full_code).strip()
                code_to_compile_list.append(code_to_compile)
            else:
                print(f"[[Warning]] Error in full_code or no code found")
                code_to_compile_list.append("")
                
        except Exception as e:
            print(f"[[Warning]] Error processing solution: {e}")
            code_to_compile_list.append("")
    
    return code_to_compile_list


def main():
    """Main function"""
    # Model path
    model_path = "/scratch/gpfs/PLI/yong/averaged_models_2/Qwen3-8B-A2_avg-0_90"
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return
    
    # Check if compiler directories exist
    if not os.path.exists(REQUEST_DIR):
        print(f"Warning: Request directory {REQUEST_DIR} does not exist. Creating...")
        os.makedirs(REQUEST_DIR, exist_ok=True)
    
    if not os.path.exists(RESPONSE_DIR):
        print(f"Warning: Response directory {RESPONSE_DIR} does not exist. Creating...")
        os.makedirs(RESPONSE_DIR, exist_ok=True)
    
    # Hardcoded input - based on Goedel-Prover-V2 format from HuggingFace
    hardcoded_input = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
""".strip()

    # Create prompt in Goedel format (exactly as shown in HuggingFace example)
    prompt = f"""
Complete the following Lean 4 code:

```lean4
{hardcoded_input}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

    # Create chat format
    chat = [
        {"role": "user", "content": prompt},
    ]
    
    print("Loading model...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load model with vLLM
    max_tokens = 1024*24
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=max_tokens
    )
    
    print("Model loaded successfully!")
    
    # Create sampling parameters
    max_gen_tokens = 1024*8
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_gen_tokens,
        n=1  # Generate 8 samples
    )
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    print("Starting inference...")
    print(f"Input prompt: {formatted_prompt[:200]}...")
    
    # Generate responses
    start_time = time.time()
    outputs = llm.generate([formatted_prompt], sampling_params)
    generation_time = time.time() - start_time
    
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Process outputs
    solution_strs = []
    print("\n" + "="*80)
    print("GENERATED SOLUTIONS:")
    print("="*80)
    
    for i, output in enumerate(outputs[0].outputs):
        # Combine input and output
        full_solution = formatted_prompt + output.text
        solution_strs.append(full_solution)
        print(f"\nSolution #{i+1}:")
        print("-"*40)
        print(f"Length: {len(output.text)} characters")
        print("\nOutput text:")
        print("-"*20)
        # Print full output with line breaks for readability
        print(output.text)
        print("-"*40)
    
    print(f"\nTotal solutions generated: {len(solution_strs)}\n")
    
    # Process solution strings to extract Lean code
    print("Processing solution strings to extract Lean code...")
    code_to_compile_list = process_solution_strings(solution_strs)
    
    print("\n" + "="*80)
    print("EXTRACTED LEAN CODE:")
    print("="*80)
    print(f"\nExtracted {len(code_to_compile_list)} code blocks")
    
    # Filter out empty code blocks
    valid_codes = [code for code in code_to_compile_list if code.strip()]
    print(f"Valid code blocks: {len(valid_codes)}")
    
    for i, code in enumerate(valid_codes):
        print(f"\nCode Block #{i+1}:")
        print("-"*40)
        print(code)
        print("-"*40)
    
    if valid_codes:
        print("\nSending codes to compiler...")
        # Send to compiler
        compiled_results = compile_lean(valid_codes)
        
        if compiled_results:
            print("\n" + "="*80)
            print("COMPILATION RESULTS:")
            print("="*80)
            print(f"\nReceived {len(compiled_results)} compilation results\n")
            
            # Analyze results
            successful_compilations = 0
            for i, result in enumerate(compiled_results):
                print(f"\nResult #{i+1}: {result['name']}")
                print("-"*40)
                
                if result['compilation_result']['pass'] and result['compilation_result']['complete']:
                    successful_compilations += 1
                    print("✅ Status: Compilation Successful")
                else:
                    print("❌ Status: Compilation Failed")
                    if 'errors' in result['compilation_result']:
                        print(f"\nErrors ({len(result['compilation_result']['errors'])}):")
                        for error in result['compilation_result']['errors']:
                            print(f"- {error}")
                print("-"*40)
            
            print(f"\nFinal Success Rate: {successful_compilations}/{len(compiled_results)}")
            print(f"Success Percentage: {successful_compilations/len(compiled_results)*100:.1f}%")
        else:
            print("\n❌ No compilation results received")
    else:
        print("\n❌ No valid code blocks to compile")
    
    print("\n" + "="*80)
    print("SCRIPT COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
