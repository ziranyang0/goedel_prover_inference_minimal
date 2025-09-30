# goedel-prover-inference minimal

This directory contains scripts for model inference and Lean code compilation.

## File Description

### 1. `inference_script.py` - Complete Inference Script
This is a complete self-contained script that includes:
- Model loading and vLLM inference
- String processing logic to extract Lean code
- Sending requests to compiler service
- Processing compilation results

### 2. `test_inference_script.py` - Test Inference Script
This is a simplified version for testing only:
- Model loading and vLLM inference
- Save results to JSON file
- No compiler integration

## Usage

### Test Inference (Recommended to run first)
```bash
python test_inference_script.py
```

### Complete Inference (with compiler)
```bash
python inference_script.py
```

## Dependencies

```bash
pip install vllm transformers torch
```

## Configuration

### Model Path
Hardcoded model path in the script:
```python
model_path = "/scratch/gpfs/PLI/yong/averaged_models_2/Qwen3-8B-A2_avg-0_90"
```

### Compiler Service
Ensure the following directories exist:
- `/home/bl3615/data/shared_a/requests` - Request directory
- `/home/bl3615/data/shared_a/responses` - Response directory

### Hardcoded Input
The script uses hardcoded input in Goedel-Prover-V2 format:
```lean4
theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
```

## Output Description

### Test Script Output
- Console displays generation process
- Saves results to `test_inference_results.json`

### Complete Script Output
- Console displays generation and compilation process
- Shows compilation success rate
- Shows successful/failed compilation results

## Parameter Tuning

### vLLM Parameters
```python
sampling_params = SamplingParams(
    temperature=0.7,    # Temperature parameter
    top_p=0.9,         # top-p sampling
    max_tokens=32768,   # Maximum tokens
    n=8                 # Number of samples to generate
)
```

### Model Parameters
```python
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=1,      # Parallelism
    gpu_memory_utilization=0.8, # GPU memory utilization
    max_model_len=32768         # Maximum model length
)
```

## Troubleshooting

1. **Model path does not exist**: Check if the model path is correct
2. **Insufficient GPU memory**: Reduce `gpu_memory_utilization` or `max_model_len`
3. **Compiler service unavailable**: Check if request/response directories exist
4. **Empty generation results**: Check input format and model configuration

## Notes

- Ensure sufficient GPU memory
- Check if compiler service is running
- Monitor generation time and memory usage
- Verify generated Lean code format
