#!/usr/bin/env python3
"""
Fine-tuned Model Integration Helper

This script provides utilities to integrate a locally fine-tuned language model
into the ATC simulator, replacing the Groq API calls with local inference.

Usage:
1. First fine-tune a model using finetune_atc_agent.py
2. Run this script to test the model and generate integration code:
   python integrate_finetuned_model.py --model_path "./finetuned_atc_model"
"""

import os
import argparse
import time
from typing import Dict, Any, Optional, Union

# Import transformers
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel, PeftConfig
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    print("Warning: transformers, torch, or peft packages not found. Install with:")
    print("pip install transformers torch peft")

# Create a wrapper class that mimics the Groq LLM interface
class LocalLLMWrapper:
    """
    A wrapper for a local language model that provides an interface similar to Groq's API.
    This allows easy integration with the ATC simulator.
    """
    
    def __init__(self, model_path: str, max_tokens: int = 512, temperature: float = 0.1):
        """
        Initialize the local LLM wrapper.
        
        Args:
            model_path: Path to the fine-tuned model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature parameter for generation (higher = more random)
        """
        if not HAVE_TRANSFORMERS:
            raise ImportError("transformers package is required for LocalLLMWrapper")
        
        print(f"Loading model from {model_path}...")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Check if it's a LoRA (PEFT) model
        peft_config_path = os.path.join(model_path, "adapter_config.json")
        is_peft_model = os.path.exists(peft_config_path)
        
        # Load model with appropriate settings
        if is_peft_model:
            # For LoRA models, need to load the base model first
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_path = peft_config.base_model_name_or_path
            
            print(f"Loading base model {base_model_path} for LoRA...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                self.model, 
                model_path,
                torch_dtype=torch.float16
            )
        else:
            # For full models, load directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set up default generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
            do_sample=(temperature > 0.01),
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        print("Model loaded and ready for inference.")
    
    def invoke(self, prompt: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Either a string or a dict with a 'content' field
            
        Returns:
            A dict with a 'content' field containing the generated text
        """
        # Handle different prompt formats
        if isinstance(prompt, dict):
            if 'content' in prompt:
                input_text = prompt['content']
            else:
                input_text = str(prompt)
        else:
            input_text = prompt
        
        # Format prompt for instruction-tuned models
        formatted_prompt = self._format_prompt(input_text)
        
        # Generate response
        start_time = time.time()
        try:
            result = self.generator(formatted_prompt)[0]['generated_text']
            # Extract only the newly generated part (after the prompt)
            generated_text = result[len(formatted_prompt):].strip()
            
            # Clean up the response
            if "### Response:" in result and "### Response:" not in formatted_prompt:
                # Extract just the response part
                generated_text = result.split("### Response:")[-1].strip()
            
            # Time the generation
            duration = time.time() - start_time
            print(f"Generated response in {duration:.2f} seconds")
            
            return {"content": generated_text}
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"content": "I'm sorry, I encountered an error while generating a response."}
    
    def _format_prompt(self, text: str) -> str:
        """Format the prompt according to the expected format for instruction-tuned models"""
        # This format works for models fine-tuned with the format in finetune_atc_agent.py
        if "### Instruction:" in text or "### Response:" in text:
            # Already formatted
            return text
        
        formatted = f"""### Instruction:
{text}

### Response:
"""
        return formatted


def generate_integration_code(model_path: str) -> str:
    """Generate code snippet for integrating the model into the ATC simulator"""
    
    integration_code = f"""
# === INTEGRATION CODE FOR atc_simulator_pygame.py ===

# 1. Replace the Groq imports with transformers imports
# FROM:
# from langchain_groq import ChatGroq
# TO:
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig  # Only needed for LoRA models

# 2. Replace the LLM initialization in _initialize_agents_and_llm method:
# FROM:
# llm = ChatGroq(temperature=0.1, model_name="llama3-70b-8192")
# TO:
model_path = "{model_path}"  # Path to your fine-tuned model

# Check if it's a LoRA (PEFT) model
peft_config_path = os.path.join(model_path, "adapter_config.json")
is_peft_model = os.path.exists(peft_config_path)

# Load model with appropriate settings
if is_peft_model:
    # For LoRA models, need to load the base model first
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_path = peft_config.base_model_name_or_path
    
    print(f"Loading base model {{base_model_path}} for LoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model, 
        model_path,
        torch_dtype=torch.float16
    )
else:
    # For full models, load directly
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id
)

# Create a wrapper that mimics the Groq interface
class LocalLLMWrapper:
    def invoke(self, prompt):
        if isinstance(prompt, dict):
            if 'content' in prompt:
                input_text = prompt['content']
            else:
                input_text = str(prompt)
        else:
            input_text = prompt
            
        # Format the prompt for instruction-tuned models
        formatted_prompt = f\"\"\"### Instruction:
{{input_text}}

### Response:
\"\"\"
        
        result = text_generator(formatted_prompt)[0]['generated_text']
        # Extract only the response part
        if "### Response:" in result:
            response_text = result.split("### Response:")[-1].strip()
        else:
            response_text = result[len(formatted_prompt):].strip()
            
        return {{"content": response_text}}

# Use this wrapper as your LLM
llm = LocalLLMWrapper()
"""
    
    return integration_code


def test_model_on_atc_scenarios(model_path: str, num_tests: int = 5) -> None:
    """Test the fine-tuned model on some ATC scenarios"""
    if not HAVE_TRANSFORMERS:
        print("Cannot test model: transformers package not installed")
        return
    
    # Create LLM wrapper
    local_llm = LocalLLMWrapper(model_path)
    
    # Sample ATC scenarios for testing
    test_scenarios = [
        "How should I sequence three aircraft approaching runway 27R with similar ETAs?",
        "Aircraft AAL123 and UAL456 are on converging paths at FL300. What actions should I take?",
        "Flight SWA789 is requesting immediate landing due to low fuel. How should I prioritize?",
        "What is the correct phraseology to instruct DAL456 to hold at BRAVO fix at 7,000 feet?",
        "How should I manage departure sequencing when runway 18L visibility drops below minimums?",
        "What's the appropriate response when a pilot reports wind shear on final approach?",
        "How should I coordinate traffic if there's a disabled aircraft on the main runway?",
        "What's the proper procedure for an aircraft declaring emergency due to engine failure?",
    ]
    
    print(f"\n===== Testing Model on {min(num_tests, len(test_scenarios))} ATC Scenarios =====\n")
    
    for i, scenario in enumerate(test_scenarios[:num_tests]):
        print(f"Scenario {i+1}: {scenario}")
        print("-" * 80)
        
        start_time = time.time()
        response = local_llm.invoke(scenario)
        duration = time.time() - start_time
        
        print(f"Response ({duration:.2f}s):")
        print(response["content"])
        print("=" * 80)
        print()


def main():
    parser = argparse.ArgumentParser(description="Fine-tuned model integration helper")
    parser.add_argument("--model_path", type=str, default="./finetuned_atc_model",
                       help="Path to the fine-tuned model")
    parser.add_argument("--test", action="store_true",
                       help="Test the model on ATC scenarios")
    parser.add_argument("--num_tests", type=int, default=5,
                       help="Number of test scenarios to run")
    parser.add_argument("--output_file", type=str, 
                       help="Output file for integration code")
    
    args = parser.parse_args()
    
    if args.test and HAVE_TRANSFORMERS:
        test_model_on_atc_scenarios(args.model_path, args.num_tests)
    
    # Generate integration code
    integration_code = generate_integration_code(args.model_path)
    
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(integration_code)
        print(f"Integration code written to {args.output_file}")
    else:
        print("\n===== Integration Code =====\n")
        print(integration_code)
    
    print("\nSteps to integrate into ATC simulator:")
    print("1. Install required packages if not already installed:")
    print("   pip install transformers torch peft")
    print(f"2. Update atc_simulator_pygame.py with the integration code")
    print("3. Run the ATC simulator as usual:")
    print("   python atc_simulator_pygame.py")
    print("\nNote: Local inference requires more memory than API calls.")


if __name__ == "__main__":
    main() 