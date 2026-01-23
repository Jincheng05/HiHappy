# HiHappy: Achieving Co-frequency in Psychological Counseling

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-yellow.svg)](https://huggingface.co/transformers/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.12+-orange.svg)](https://www.deepspeed.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Choose your language**

[![中文文档](https://img.shields.io/badge/文档-中文版-red?style=flat-square&logo=readme&logoColor=white)](README_ZH.md)
&nbsp;&nbsp;&nbsp;&nbsp;
[![English Docs](https://img.shields.io/badge/docs-English-blue?style=flat-square&logo=readme&logoColor=white)](README.md)

*LLM-based Psychological Counseling Dialogue System - Achieving Emotional Synchronization and Root Cause Exploration*


</div>

---

## 📖 Introduction

### Background and Motivation

The core of psychological counseling lies in establishing an effective counseling relationship, and **co-frequency ability** is the key to achieving this goal. Traditional psychological counseling dialogue systems often overlook emotional synchronization, communication rhythm matching, and implicit need understanding between counselors and clients, leading to poor dialogue quality.

<!-- 
📸 Image Position 1: System Overview
Filename: assets/overview.png
Description: Display the overall architecture and core functions of HiHappy system
Recommended size: 1200x600px
-->
<div align="center">
  <img src="assets/overview.png" alt="HiHappy System Overview" width="800"/>
  <p><i>Figure 1: HiHappy Psychological Counseling Dialogue System Architecture</i></p>
</div>

This project implements the innovative methods proposed in the paper "HiHappy: Achieving Co-frequency Alignment and Positive Emotional Shift in Psychological Counseling LLMs", solving these problems through the following technical breakthroughs:

1. **Multi-dimensional Evaluation System**: Established a psychological counseling dialogue evaluation framework with 8 core dimensions
2. **Four-LLM Collaborative Generation**: Innovatively uses counselor, client, summarizer, and evaluator models to collaboratively generate high-quality training data
3. **Dynamic Strategy Adjustment**: Dynamically adjusts counseling strategies and rhythm based on real-time emotion analysis
4. **Professional Technique Integration**: Integrates multiple psychological counseling techniques such as REBT, CBT, and Humanistic therapy

### Core Capabilities

This system achieves the following core psychological counseling capabilities through fine-tuning large language models (Qwen3-8B):

| Capability Dimension | Specific Performance | Technical Implementation |
|---------------------|---------------------|-------------------------|
| **Co-frequency Ability** | Accurately capture client's emotional state, match communication rhythm, respect boundaries, understand implicit needs | Four-dimensional evaluation (Emotion Capture 40% + Communication Rhythm 30% + Boundary Respect 20% + Implicit Needs 10%) |
| **Relationship Building** | Build safe, trusting, collaborative, and stable counseling relationships | Trust building, acceptance and safety, interaction adaptation, authenticity assessment |
| **Dialogue Strategy** | Adopt appropriate psychological counseling techniques (REBT, CBT, Humanistic, SFBT, EFT, etc.) | Strategy adaptability, effectiveness, flexibility, ethical compliance |
| **Emotional Empathy** | Feel and respond to client's emotional experiences | Emotional awareness, response adaptation, resonance depth, non-invasiveness |
| **Emotional Improvement** | Effectively alleviate client's negative emotions and enhance positive emotions | Emotional change trends, improvement magnitude, stability assessment |
| **Cognitive Empathy** | Understand client's thoughts, perspectives, and cognitive patterns | Perspective understanding, cognitive restructuring, guidance effectiveness |
| **State and Attitude** | Maintain professional, stable, and positive counseling state | Professionalism, stability, positivity assessment |

### 2. High-Quality Data Generation Mechanism

#### 2.1 Four-LLM Collaborative Generation Architecture

<!-- 
📸 Image Position 3: Four-LLM Collaborative Architecture
Filename: assets/three-llm-architecture.png
Description: Display the collaborative workflow of counselor, client, summarizer, and evaluator models
Recommended size: 800x1000px (vertical)
-->
<div align="center">
  <img src="assets/Multi-agent-architecture.png" alt="Four-LLM Collaborative Architecture" width="800"/>
  <p><i>Figure 3: Four-LLM Collaborative Generation Architecture - Counselor, Client, Summarizer, and Evaluator Models Working Together</i></p>
</div>

The summarizer model analyzes the client's emotional state and outputs a seven-dimensional emotion vector:

```python
Emotion Vector = [
    Happy: 0.15,           # Positive emotion
    Calm: 0.70,            # Positive emotion
    Anxious: 0.30,         # Negative emotion
    Sad: 0.10,             # Negative emotion
    Angry: 0.05,           # Negative emotion
    Guilt/Shame: 0.10,     # Negative emotion
    Helpless/Hopeless: 0.05 # Negative emotion
]
```

**Emotion Cause Analysis**:
- **Positive Emotion Causes**: The counselor's empathetic response acknowledged their efforts in caring for family members, making them feel understood, and increasing their sense of calm
- **Negative Emotion Causes**: Long-term care for sick family members leads to sleep deprivation, and concerns about their own health status produce persistent anxiety and mild helplessness

#### 2.3 Dynamic Strategy Adjustment Mechanism

The counselor model dynamically adjusts strategies based on the summarizer's analysis:

**When Emotions Are Extremely Negative** (e.g., Helpless/Hopeless ≥ 0.7):
- Prioritize humanistic empathetic acceptance techniques
- Avoid direct cognitive debate interventions
- Slow down the pace and increase listening

**When Emotions Gradually Improve** (Calm + Happy ≥ 0.8):
- Timely introduce REBT/CBT cognitive debate techniques
- Guide clients to identify irrational cognitions
- Advance cognitive restructuring

**When Emotions Are Stable** (Calm ≥ 0.6 and Negative Emotions ≤ 0.3):
- Combine solution-focused brief therapy to advance problem-solving
- Consolidate rational cognition
- Develop action plans

### 3. Model Training Optimization

#### 3.1 LoRA Efficient Fine-tuning

**LoRA Configuration Details**:

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    target_modules=[
        "q_proj",      # Query projection layer
        "k_proj",      # Key projection layer
        "v_proj",      # Value projection layer
        "o_proj",      # Output projection layer
        "gate_proj",   # FFN gate layer
        "up_proj",     # FFN up projection layer
        "down_proj"    # FFN down projection layer
    ],
    
    inference_mode=False,  # Training mode
    r=16,                  # LoRA rank
    lora_alpha=32,        # LoRA scaling factor
    lora_dropout=0.05,    # Dropout ratio
)
```

#### 3.2 DeepSpeed Distributed Training

**ZeRO-2 Optimization Strategy**:

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

## 🚀 Quick Start

### Software Requirements

```bash
Operating System: Linux (Ubuntu 20.04+ recommended)
Python: 3.8 - 3.11
CUDA: 11.8 or 12.1
cuDNN: 8.6+
```

### Environment Setup

#### Method 1: Using Conda (Recommended)

```bash
# 1. Create virtual environment
conda create -n hihappy python=3.10
conda activate hihappy

# 2. Install PyTorch (choose based on CUDA version)
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Clone repository
git clone https://github.com/yourusername/HiHappy.git
cd HiHappy

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install DeepSpeed
pip install deepspeed

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🔧 Model Training

### Method 1: Using Provided Training Script

```bash
# Single GPU training
python train_Q3-8b.py

# Multi-GPU training (using DeepSpeed)
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 train_Q3-8b-Emo+four_data.py
```

**Training Configuration Instructions**:

Edit `train_Q3-8b.py` to modify the following configurations:

```python
# Model path
model_name = "/path/to/Qwen3-8B-Instruct"

# Dataset path
data_files = [
    "/path/to/train_data.json",
]

# Output path
output_dir = "/path/to/output/lora_model"

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
```

### Method 2: Using LLaMA-Factory Training (Recommended)

This project provides a complete LLaMA-Factory configuration file `training_args.yaml`, which can be used directly for training with LLaMA-Factory.

#### Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

#### Configure Dataset

Add your dataset in `LLaMA-Factory/data/dataset_info.json`:

```json
{
  "my_counseling_data": {
    "file_name": "/path/to/your/train_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages"
    }
  }
}
```

#### Train Using Configuration File

```bash
# Copy configuration file to LLaMA-Factory directory
cp training_args.yaml LLaMA-Factory/

# Start training
cd LLaMA-Factory
llamafactory-cli train training_args.yaml
```

**Configuration File Description** (`training_args.yaml`):

```yaml
# Model configuration
model_name_or_path: /path/to/Qwen3-8B-Instruct
output_dir: saves/Qwen3-8B-Instruct/lora/train

# Training parameters
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-05
num_train_epochs: 2.0

# LoRA configuration
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_target: all

# DeepSpeed configuration
deepspeed: ds_z2_offload_config.json

# Dataset configuration
dataset: my_counseling_data
dataset_dir: data
cutoff_len: 8192
```

### Training Monitoring

The training process will be automatically logged to SwanLab. Visit https://swanlab.cn to view:
- Loss curves
- Learning rate changes
- GPU utilization
- Training speed

## 📊 Model Evaluation

After training, you can use the evaluation scripts provided in the project for comprehensive model assessment.

### Evaluation Dimensions

The project provides evaluation scripts for 7 core dimensions:

| Dimension | Script File | Description |
|-----------|-------------|-------------|
| Co-frequency Ability | `model_evaluation/cofrequency_evaluation.py` | Evaluate emotion capture, communication rhythm, boundary respect, implicit needs |
| Relationship Building | `model_evaluation/relationship_evaluation.py` | Evaluate trust building, safety creation, acceptance expression |
| Dialogue Strategy | `model_evaluation/strategy_evaluation.py` | Evaluate application of REBT, CBT and other counseling techniques |
| Emotional Empathy | `model_evaluation/emotional_empathy_evaluation.py` | Evaluate emotional awareness, response adaptation, resonance depth |
| Emotion Improvement | `model_evaluation/emotion_improvement_evaluation.py` | Evaluate emotion change trends, improvement magnitude, stability |
| Cognitive Empathy | `model_evaluation/cognitive_empathy_evaluation.py` | Evaluate perspective understanding, cognitive restructuring, guidance effectiveness |
| State & Attitude | `model_evaluation/state_attitude_evaluation.py` | Evaluate professionalism, stability, positivity |

### Running Evaluation

#### 1. Start Model Inference Service

First, start the vLLM inference service:

```bash
# Start vLLM service (using your fine-tuned model)
python -m vllm.serve.openai \
    --model /path/to/your/finetuned/model \
    --port 6006 \
    --gpu-memory-utilization 0.9
```

#### 2. Run Single Dimension Evaluation

```bash
# Evaluate co-frequency ability
python model_evaluation/cofrequency_evaluation.py

# Evaluate emotional empathy
python model_evaluation/emotional_empathy_evaluation.py

# Evaluate other dimensions...
```

#### 3. Batch Evaluate All Dimensions

```bash
# Run all evaluation scripts sequentially
for script in model_evaluation/*_evaluation.py; do
    python "$script"
done
```

#### 4. Calculate Overall Score

```bash
# Calculate average scores across all dimensions
python model_evaluation/calculate_average_score.py
```

### Evaluation Configuration

Before running evaluations, modify the configuration in each script:

```python
# Configuration example
VLLM_BASE_URL = "http://localhost:6006/v1"  # vLLM service address
VLLM_MODEL_NAME = "/path/to/your/model"     # Model path
DATASET_PATH = "/path/to/test_data.json"    # Test dataset path
OUTPUT_PATH = "/path/to/output/results.json" # Output path
MAX_CONCURRENCY = 10                         # Max concurrency
```

### Dataset Evaluation

To evaluate the quality of training dataset annotations, use scripts in `dataset_evaluation/`:

```bash
# Evaluate co-frequency ability annotations in dataset
python dataset_evaluation/cofrequency_evaluation.py

# Evaluate other dimensions...
```

### Evaluation Results

After evaluation, JSON format result files will be generated, containing:
- Detailed scores for each sample (0-3 points)
- Evaluation analysis
- Average score statistics
- Success rate statistics

Example output:
```json
[
  {
    "Sample ID": "sample_001",
    "Dialog ID": "dialog_001",
    "Turn Number": 1,
    "Co-frequency Score": 3,
    "Analysis": "The model accurately captured the client's anxiety..."
  }
]
```

## 📝 Citation

If you use the code or methods from this project, please cite our paper:

```bibtex
@article{hihappy2024,
  title={HiHappy: Achieving Co-frequency in Psychological Counseling},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 🤝 Contributing

Issues and Pull Requests are welcome!

### Contribution Guidelines

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📚 References

### Paper

- [HiHappy: Achieving Co-frequency in Psychological Counseling](984_HiHappy_Achieving_Co_frequ%20(3).pdf)

### Related Work

- **Psychological Counseling Theory**:
  - Ellis, A. (1962). Reason and emotion in psychotherapy. (REBT Theory)
  - Beck, A. T. (1979). Cognitive therapy and the emotional disorders. (CBT Theory)
  - Rogers, C. R. (1951). Client-centered therapy. (Humanistic Theory)

- **Large Language Models**:
  - Qwen Team. (2024). Qwen2.5 Technical Report.
  - Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.

- **Psychological Counseling Dialogue Systems**:
  - Liu, Z., et al. (2021). Towards Empathetic Open-domain Conversation Models.
  - Sharma, A., et al. (2020). A Computational Approach to Understanding Empathy.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

```
MIT License

Copyright (c) 2024 HiHappy Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/HiHappy?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/HiHappy?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/HiHappy?style=social)
![GitHub contributors](https://img.shields.io/github/contributors/yourusername/HiHappy)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/HiHappy)
![GitHub issues](https://img.shields.io/github/issues/yourusername/HiHappy)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/HiHappy)

---

<div align="center">

**⭐ If this project helps you, please give us a Star! ⭐**

**Making AI More Understanding, Making Psychological Counseling More Warm**

Made with ❤️ by HiHappy Team

[Back to Top](#hihappy-achieving-co-frequency-in-psychological-counseling)

</div>
