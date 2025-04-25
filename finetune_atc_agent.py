#!/usr/bin/env python3
"""
ATC Agent Fine-tuning Script

This script fine-tunes a language model on ATC-specific data to create a specialized 
agent for the ATC simulator. The fine-tuned model should reduce hallucinations
and improve domain-specific reasoning for air traffic control scenarios.

Requirements:
- transformers
- datasets
- peft
- bitsandbytes (for quantization)
- trl (for training with RLHF components)
- accelerate
- torch

Usage:
python finetune_atc_agent.py --model_name "meta-llama/Llama-2-7b-hf" --output_dir "./finetuned_atc_model"
"""

import os
import json
import argparse
import random
import torch
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from trl import SFTTrainer

# Ensure reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ATC domain-specific prompts and responses
def generate_atc_dataset(num_samples: int = 1000) -> List[Dict[str, str]]:
    """Generate synthetic ATC instruction data for fine-tuning"""
    atc_scenarios = [
        # Conflict resolution scenarios
        "Two aircraft are on converging paths at the same altitude. How would you resolve this conflict?",
        "Flight AAL123 is approaching runway 27R while UAL456 is still on the runway. What action should you take?",
        "An aircraft reports severe turbulence at FL300. Several aircraft are requesting that altitude. What would you do?",
        
        # Landing and takeoff management 
        "There are 5 aircraft in the landing sequence and a new emergency aircraft calls in. How do you prioritize?",
        "An aircraft aborts takeoff due to a technical issue. How do you manage the departure sequence?",
        "Runway 09L is closed due to maintenance. How do you reorganize the arrival sequence?",
        
        # Weather-related scenarios
        "A thunderstorm is moving toward the airport from the west. How do you adapt your traffic management?",
        "Visibility is rapidly decreasing. What actions should you take for aircraft on approach?",
        "Strong crosswinds are reported at the airport. How would you advise pilots and adjust operations?",
        
        # Radio communications
        "How would you properly phrase a clearance for an aircraft to descend to 10,000 feet?",
        "What is the correct phraseology to clear an aircraft for ILS approach to runway 27L?",
        "How should you inform an aircraft about traffic in their vicinity?",
        
        # Unusual situations
        "An aircraft declares emergency due to engine failure. What's your immediate response?",
        "A pilot reports a drone sighting near the approach path. What actions should you take?",
        "Two aircraft call in simultaneously on the same frequency. How do you manage this situation?",
        
        # Decision making with limited resources
        "All runways except one are closed due to snow. How do you prioritize traffic?",
        "You have multiple aircraft requesting immediate descent due to turbulence. How do you sequence them?", 
        "The airport is at capacity with ground holds. A flight declares minimum fuel. What do you do?",
    ]
    
    # Extend with parametrized scenarios
    callsigns = ["AAL123", "UAL456", "DAL789", "SWA456", "JBU234", "FFT567", "ASA890", "BAW123"]
    runways = ["27R", "09L", "18C", "36L", "27L", "09R", "18L", "36C"]
    altitudes = ["FL240", "FL300", "FL350", "10,000 feet", "5,000 feet", "7,000 feet", "12,000 feet"]
    fix_points = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "HOTEL", "ZULU"]
    
    for _ in range(len(atc_scenarios), num_samples):
        scenario_type = random.choice([
            "conflict", "landing", "takeoff", "weather", "emergency", "communication"
        ])
        
        if scenario_type == "conflict":
            callsign1 = random.choice(callsigns)
            callsign2 = random.choice([c for c in callsigns if c != callsign1])
            altitude = random.choice(altitudes)
            atc_scenarios.append(
                f"Aircraft {callsign1} and {callsign2} are approaching the same fix at {altitude}. How would you resolve this conflict?"
            )
        elif scenario_type == "landing":
            callsign = random.choice(callsigns)
            runway = random.choice(runways)
            atc_scenarios.append(
                f"Flight {callsign} is cleared to land on runway {runway}, but reports unstable approach. What instructions would you give?"
            )
        elif scenario_type == "takeoff":
            callsign = random.choice(callsigns)
            runway = random.choice(runways)
            atc_scenarios.append(
                f"Flight {callsign} is number 3 for takeoff on runway {runway}. How would you sequence and provide clearance?"
            )
        elif scenario_type == "weather":
            condition = random.choice(["fog", "thunderstorm", "snow", "wind shear", "hail"])
            runway = random.choice(runways)
            atc_scenarios.append(
                f"A {condition} is affecting approaches to runway {runway}. How would you manage the traffic?"
            )
        elif scenario_type == "emergency":
            callsign = random.choice(callsigns)
            emergency = random.choice(["engine failure", "medical emergency", "cabin pressure issue", "bird strike", "hydraulic failure"])
            atc_scenarios.append(
                f"Flight {callsign} declares an emergency due to {emergency}. What actions would you take?"
            )
        elif scenario_type == "communication":
            callsign = random.choice(callsigns)
            fix = random.choice(fix_points)
            altitude = random.choice(altitudes)
            atc_scenarios.append(
                f"What is the correct phraseology to instruct {callsign} to proceed to {fix} and maintain {altitude}?"
            )
    
    # Format the dataset for instruction fine-tuning
    formatted_data = []
    for i, instruction in enumerate(atc_scenarios[:num_samples]):
        formatted_data.append({
            "instruction": instruction,
            "input": "",  # Empty for pure instruction style
            "output": f"[This would be expert ATC response {i+1}]"  # In a real dataset, this would be filled with expert responses
        })
    
    return formatted_data

def create_atc_dataset(dataset_path: str = None, num_samples: int = 1000) -> Dataset:
    """Create or load a dataset for fine-tuning"""
    if dataset_path and os.path.exists(dataset_path):
        # Load existing dataset
        try:
            dataset = load_dataset(dataset_path)
            print(f"Loaded existing dataset from {dataset_path}")
            return dataset['train']
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Generating synthetic dataset instead.")
    
    # Generate synthetic dataset
    data = generate_atc_dataset(num_samples)
    dataset = Dataset.from_list(data)
    
    # Save dataset for future use
    if dataset_path:
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        dataset.save_to_disk(dataset_path)
        print(f"Saved synthetic dataset to {dataset_path}")
    
    return dataset

def format_instruction(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Format the instruction examples for the model"""
    # This format works well for Llama models
    instruction_template = """### Instruction:
{instruction}

{input}

### Response:
{output}
"""
    
    formatted_text = instruction_template.format(
        instruction=example["instruction"],
        input=example["input"] if example["input"] else "",
        output=example["output"]
    )
    
    example["text"] = formatted_text
    return example

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for ATC agent")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", 
                        help="Base model to fine-tune (default: meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--output_dir", type=str, default="./finetuned_atc_model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--dataset_path", type=str, default="./atc_dataset",
                        help="Path to load/save the dataset")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of training samples to generate if dataset doesn't exist")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print(f"Loading base model: {args.model_name}")
    
    # Load base model and tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare the model with LoRA for parameter-efficient fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    # Create or load dataset
    dataset = create_atc_dataset(args.dataset_path, args.num_samples)
    
    # Format dataset with instructions
    dataset = dataset.map(
        lambda example: format_instruction(example, tokenizer),
        remove_columns=dataset.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="tensorboard",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )
    
    # Initialize the SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=512,
    )
    
    # Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.output_dir)
    print(f"Model fine-tuning complete. Saved to {args.output_dir}")
    
    # Instructions for using the fine-tuned model
    print("\nTo use this model in the ATC simulator:")
    print("1. Load the model in atc_simulator_pygame.py by changing the ChatGroq initialization to:")
    print("   from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline")
    print(f"   model_path = '{args.output_dir}'")
    print("   model = AutoModelForCausalLM.from_pretrained(model_path)")
    print("   tokenizer = AutoTokenizer.from_pretrained(model_path)")
    print("   # Create a pipeline for text generation")
    print("   llm = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=512)")
    print("2. Then adapt the agent cycle to use this local model instead of Groq.")

if __name__ == "__main__":
    main() 