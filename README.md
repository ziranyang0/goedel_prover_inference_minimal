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

### Conda Environment Setup (Recommended)
Based on [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html):

```bash
# Create conda environment (Python >= 3.10 required for verl)
conda create -n verl-inference python=3.10 -y
conda activate verl-inference

# Install verl dependencies using the provided script
# For FSDP backend only (recommended for inference)
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# Or for full verl with Megatron support
bash scripts/install_vllm_sglang_mcore.sh

# Install verl from source
git clone https://github.com/volcengine/verl.git
cd verl
pip install --no-deps -e .
```

### System Requirements
- **Python**: Version >= 3.10
- **CUDA**: Version >= 12.1 (recommended >= 12.4)
- **cuDNN**: Version >= 9.8.0

### Alternative: Direct pip installation (Not recommended)
```bash
pip install vllm transformers torch
```

## Configuration

### Model Path
The model path can be configured in the script. You have several options:

#### Option 1: Local Model (Current Configuration)
```python
model_path = "/scratch/gpfs/PLI/yong/averaged_models_2/Qwen3-8B-A2_avg-0_90"
```
This is a local 8B SFT model before RL training.

#### Option 2: Hugging Face Models
You can use any model from Hugging Face, for example:
```python
# Goedel-Prover-V2-8B (recommended for theorem proving)
model_path = "Goedel-LM/Goedel-Prover-V2-8B"

# Or other models
model_path = "Goedel-LM/Goedel-Prover-V2-32B"
```

#### Option 3: Custom Model Path
Replace with your own model path:
```python
model_path = "/path/to/your/model"
```

**Note**: When using Hugging Face models, the script will automatically download the model on first run. Make sure you have sufficient disk space and internet connection.

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
5. **Conda environment issues**: 
   - Ensure conda is properly installed
   - Activate the environment: `conda activate verl-inference`
   - Check Python version: `python --version` (should be >= 3.10)
   - Verify CUDA installation: `nvidia-smi`
6. **Package installation failures**:
   - Update conda: `conda update conda`
   - Clear conda cache: `conda clean --all`
   - Try installing packages individually
7. **verl installation issues**:
   - Ensure CUDA >= 12.1 is installed
   - Check if verl installation script completed successfully
   - Verify vLLM installation: `python -c "import vllm"`
   - For vLLM issues, try: `VLLM_USE_V1=1` environment variable
8. **Dependency conflicts**:
   - Use fresh conda environment as recommended by verl
   - Install inference frameworks first before other packages
   - Check package versions: `pip list | grep torch`

## Notes

- Ensure sufficient GPU memory
- Check if compiler service is running
- Monitor generation time and memory usage
- Verify generated Lean code format
- Always activate conda environment before running scripts: `conda activate verl-inference`
- Ensure CUDA is properly installed and accessible
- For optimal vLLM performance, set `VLLM_USE_V1=1` environment variable
- Follow verl installation guide for proper dependency management
- Use fresh conda environment to avoid package conflicts
- Refer to [verl documentation](https://verl.readthedocs.io/en/latest/start/install.html) for detailed installation instructions
