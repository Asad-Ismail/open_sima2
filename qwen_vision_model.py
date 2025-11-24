"""
Unsloth Qwen Vision Model with Instructor for Structured Output
Exports the working Unsloth Qwen model from notebook with structured output support
"""

from unsloth import FastVisionModel
import torch
from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal
import json
import re


class GameAction(BaseModel):
    """Structured output for game actions"""
    reasoning: str = Field(description="The agent's reasoning about the current game state")
    visibility_status: Literal["hidden", "visible", "partially_visible"] = Field(
        description="Current visibility status based on the Light Gem"
    )
    detected_threats: str = Field(description="Any guards or threats detected in the scene")
    recommended_action: Literal["w", "s", "a", "d", "c", "space", "enter", "none"] = Field(
        description="The recommended control action to take"
    )
    action_explanation: str = Field(description="Brief explanation of why this action is recommended")


class QwenVisionAgent:
    """Qwen Vision Model Agent with structured output using Instructor"""
    
    def __init__(self, model_name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"):
        """
        Initialize the Qwen Vision model
        
        Args:
            model_name: HuggingFace model name/path
        """
        print(f"Loading model: {model_name}")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,  # Use 4bit to reduce memory use
            use_gradient_checkpointing="unsloth",  # For long context
        )
        
        # Enable inference mode
        FastVisionModel.for_inference(self.model)
        
        self.system_prompt = """You are an elite stealth agent playing 'The Dark Mod'.
Your goal is to navigate the level, steal loot, and avoid detection.

CONTROLS YOU CAN USE:
- 'w': Move Forward
- 's': Move Backward
- 'a': Strafe Left
- 'd': Strafe Right
- 'c': Toggle Crouch (Use this when the Light Gem is bright/white)
- 'space': Jump / Mantle
- 'enter': Interact / Open Door / Pick up Loot

HUD GUIDE:
- Bottom Center Icon: Light Gem.
  * Dark/Black = You are hidden (Safe).
  * Bright/White = You are visible (Danger).
- Red Bar: Health.
- Compass: Direction.

CRITICAL: You MUST respond with ONLY a valid JSON object in this exact format:
{
  "reasoning": "your detailed analysis of the current game state",
  "visibility_status": "hidden" or "visible" or "partially_visible",
  "detected_threats": "description of any guards or threats",
  "recommended_action": "w" or "s" or "a" or "d" or "c" or "space" or "enter" or "none",
  "action_explanation": "brief explanation of why this action is recommended"
}

Do NOT include any text before or after the JSON object. Output ONLY valid JSON."""
    
    def analyze_frame(
        self,
        image: Image.Image,
        instruction: str = "Analyze the current game state and decide the next move.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        min_p: float = 0.1
    ) -> GameAction:
        """
        Analyze a game frame and return structured action
        
        Args:
            image: PIL Image of the game frame
            instruction: Additional instruction for the model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            min_p: Minimum probability for sampling
            
        Returns:
            GameAction: Structured action with reasoning
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # Prepare inputs
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p,
        )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        
        # Parse JSON response
        try:
            # Clean up the response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            # Try to find JSON in the response using regex
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return GameAction(**data)
            
            # If no JSON found via regex, try parsing the entire cleaned response
            data = json.loads(cleaned_response)
            return GameAction(**data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse JSON output: {e}")
            print(f"Raw response: {response[:500]}...")  # Print first 500 chars
            # Try text parsing fallback
            return self._parse_text_response(response)
    
    def simple_generate(
        self,
        image: Image.Image,
        instruction: str = "Analyze the current game state and decide the next move.",
        max_new_tokens: int = 128,
        temperature: float = 1.5,
        min_p: float = 0.1
    ) -> str:
        """
        Generate unstructured text response (without instructor)
        
        Args:
            image: PIL Image of the game frame
            instruction: Additional instruction for the model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            min_p: Minimum probability for sampling
            
        Returns:
            str: Generated text response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True
        )
        
        # Prepare inputs
        inputs = self.tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            min_p=min_p,
        )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        
        return response
    
    def _parse_text_response(self, response: str) -> GameAction:
        """
        Parse unstructured text response into GameAction
        Fallback method when JSON parsing fails
        """
        # Default values
        reasoning = response
        visibility_status = "hidden"
        detected_threats = "None detected"
        recommended_action = "none"
        action_explanation = "Based on analysis"
        
        # Try to extract action from common patterns
        action_match = re.search(r'[Aa]ction[:\s]+(\w)', response)
        if action_match:
            action = action_match.group(1).lower()
            if action in ['w', 's', 'a', 'd', 'c']:
                recommended_action = action
        
        # Check for visibility keywords
        if any(word in response.lower() for word in ['bright', 'visible', 'light', 'white']):
            visibility_status = "visible"
        elif any(word in response.lower() for word in ['partially', 'somewhat']):
            visibility_status = "partially_visible"
        
        # Check for threats
        if any(word in response.lower() for word in ['guard', 'enemy', 'patrol', 'threat']):
            detected_threats = "Guard or threat detected"
        
        return GameAction(
            reasoning=reasoning,
            visibility_status=visibility_status,
            detected_threats=detected_threats,
            recommended_action=recommended_action,
            action_explanation=action_explanation
        )


def main():
    """Example usage"""
    # Initialize the agent
    agent = QwenVisionAgent()
    
    # Load an image
    image_path = "imgs/example_darkmod.png"
    image = Image.open(image_path)
    
    print(f"\nAnalyzing image: {image_path}")
    print("=" * 80)
    
    # Get structured action
    action = agent.analyze_frame(image)
    
    print("\n STRUCTURED OUTPUT:")
    print(f"Reasoning: {action.reasoning}")
    print(f"Visibility: {action.visibility_status}")
    print(f"Threats: {action.detected_threats}")
    print(f"Action: {action.recommended_action}")
    print(f"Explanation: {action.action_explanation}")
    
    # You can also get unstructured output if needed
    # print("\n UNSTRUCTURED OUTPUT:")
    # text_response = agent.simple_generate(image)
    # print(text_response)


if __name__ == "__main__":
    main()
