"""
Vision Agent for game control using Qwen Vision Language Model.
Provides simple action sequence predictions based on goal instructions.
"""

from unsloth import FastVisionModel
import torch
from PIL import Image
import json
import re


class VisionAgent:
    """Vision Language Model agent for game control."""
    
    def __init__(self, model_path: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"):
        """
        Initialize the vision model.
        
        Args:
            model_path: Path to base model or fine-tuned model
                       Use "models/gameplay_agent_lora" for fine-tuned model
        """
        print(f"Loading model: {model_path}")
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        FastVisionModel.for_inference(self.model)
        
        self.current_goal = "Navigate to lantern"
        print(f"✓ Model loaded. Current goal: {self.current_goal}")
    
    def predict_actions(
        self,
        image: Image.Image,
        max_new_tokens: int = 64,
        temperature: float = 0.5
    ) -> list:
        """
        Predict next 7 actions given current frame and goal.
        Matches training format exactly.
        
        Args:
            image: PIL Image of the game frame
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of 7 actions (e.g. ["w", "right", "w", "w", "none", "none", "none"])
        """
        # Match training format exactly
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"Goal: {self.current_goal}\n\nPredict the next 7 actions:"}
                ]
            }
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        
        return self._parse_actions(response)
    
    def _parse_actions(self, response: str) -> list:
        """
        Parse action list from model response.
        Expected format: {"actions": ["w", "right", "w", ...]}
        """
        try:
            # Clean markdown code blocks
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "actions" in data:
                    actions = data["actions"]
                    # Ensure we have exactly 7 actions
                    if len(actions) < 7:
                        actions.extend(["none"] * (7 - len(actions)))
                    return actions[:7]
            
            # Fallback: return safe default
            print(f"Warning: Could not parse actions from: {response[:100]}")
            return ["none"] * 7
            
        except Exception as e:
            print(f"Error parsing actions: {e}")
            return ["none"] * 7
    
    def update_goal(self, new_goal: str):
        """Update the agent's goal dynamically."""
        self.current_goal = new_goal
        print(f"🎯 Goal updated to: {self.current_goal}")


def main():
    """Example usage."""
    agent = VisionAgent()
    
    # Test with an image
    image_path = "imgs/example_darkmod.png"
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return
    
    image = Image.open(image_path)
    
    print(f"\nTesting with image: {image_path}")
    print(f"Goal: {agent.current_goal}")
    print("=" * 80)
    
    actions = agent.predict_actions(image)
    
    print(f"\nPredicted actions: {actions}")
    print("=" * 80)


if __name__ == "__main__":
    import os
    main()
