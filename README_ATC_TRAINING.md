# ATC Simulator Agent Training Suite

This suite provides tools for training specialized Air Traffic Control (ATC) agents using instruction fine-tuning on domain-specific data. These agents can then be integrated into the ATC simulator to provide more reliable, less hallucinated responses.

## Overview

The training suite consists of three main components:

1. **ATC Simulator** (`atc_simulator_pygame.py`) - The main simulation environment
2. **Fine-tuning Script** (`finetune_atc_agent.py`) - For training custom LLM agents
3. **Integration Helper** (`integrate_finetuned_model.py`) - For using fine-tuned models in the simulator

## Why Fine-tune Models for ATC?

API-based LLMs like those from Groq can occasionally:
- Hallucinate ATC procedures that don't exist
- Misunderstand air traffic terminology 
- Generate responses that are too verbose for time-critical ATC contexts
- Be inconsistent in their application of procedures

By fine-tuning a model on ATC-specific data, we can:
- Reduce hallucinations and improve factual accuracy
- Make responses more concise and procedurally correct
- Ensure consistent application of ATC standards
- Have a fully local solution without API dependencies

## Workflow

### Step 1: Collect Training Data

Start by collecting a dataset of ATC-specific instructions and expert responses. This could include:

- Recordings/transcripts of actual ATC communications
- Standard operating procedures (SOPs) and training manuals
- Simulated ATC scenarios and expert solutions

For best results, format the data as instruction-response pairs.

### Step 2: Fine-tune the Model

Use the `finetune_atc_agent.py` script to fine-tune a base language model:

```bash
python finetune_atc_agent.py \
  --model_name "meta-llama/Llama-2-7b-hf" \
  --output_dir "./finetuned_atc_model" \
  --num_samples 1000 \
  --batch_size 4 \
  --num_train_epochs 3
```

This script:
- Generates or loads a dataset of ATC instructions and responses
- Quantizes the base model to 4-bit precision for memory efficiency
- Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to adapt the model
- Saves the resulting model to the specified output directory

### Step 3: Test the Fine-tuned Model

Use the integration helper script to test the model on ATC scenarios:

```bash
python integrate_finetuned_model.py \
  --model_path "./finetuned_atc_model" \
  --test \
  --num_tests 5
```

This will run the model on sample ATC scenarios to evaluate its performance.

### Step 4: Integrate with the ATC Simulator

Use the integration helper to generate code for integrating your model:

```bash
python integrate_finetuned_model.py \
  --model_path "./finetuned_atc_model" \
  --output_file "integration_code.py"
```

Then follow the instructions to update `atc_simulator_pygame.py` with the generated code.

## System Requirements

Fine-tuning a model requires significant computational resources:
- NVIDIA GPU with 8+ GB VRAM (16+ GB recommended)
- 16+ GB system RAM
- 100+ GB disk space (for model weights and datasets)

For inference only (running the fine-tuned model):
- NVIDIA GPU with 6+ GB VRAM
- 8+ GB system RAM

## Required Packages

Install the necessary packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft bitsandbytes trl accelerate
pip install pygame python-dotenv langchain
```

## Customizing the Training Process

You can customize the fine-tuning process by:

1. **Creating custom datasets**: Modify the `generate_atc_dataset` function in `finetune_atc_agent.py`
2. **Adjusting hyperparameters**: Change learning rate, epochs, batch size, etc.
3. **Using different base models**: Try Llama-3, Mistral, or other open models
4. **Modifying the instruction template**: Change the format in `format_instruction`

## Troubleshooting

- **Out of memory errors**: Reduce batch size, use smaller models, or increase quantization
- **Slow training**: Adjust gradient accumulation steps or use a smaller model
- **Poor performance**: Improve dataset quality, adjust training parameters, or use a larger base model

## References

- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft)
- [Instruction Fine-Tuning](https://huggingface.co/blog/instruction-tuning)
- [QLoRA: Efficient Fine-tuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 