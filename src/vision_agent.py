"""
Vision Agent for game control using Qwen Vision Language Model.
Provides structured analysis of game frames and action recommendations.
"""

from unsloth import FastVisionModel
import torch
from PIL import Image
from pydantic import BaseModel, Field
from typing import Literal
import json
import re


class GameAction(BaseModel):
    """Structured output for game actions."""
    reasoning: str = Field(description="Agent's reasoning about the current game state")
    visibility_status: Literal["hidden", "visible", "partially_visible"] = Field(
        description="Current visibility status based on the Light Gem"
    )
    detected_threats: str = Field(description="Any guards or threats detected")
    facing_target: bool = Field(description="True if facing the goal object")
    mouse_look: dict = Field(
        description="Mouse movement for camera control. Format: {'x': -100 to 100, 'y': -50 to 50}"
    )
    action_sequence: list[Literal["w", "s", "a", "d", "c", "space", "enter", "none"]] = Field(
        description="List of 0-5 actions to execute in sequence",
        min_length=0,
        max_length=5
    )
    action_explanation: str = Field(description="Brief explanation of recommended actions")
    goal_reached: bool = Field(description="True if goal has been reached")
    goal_status: str = Field(description="Description of progress towards goal")


class VisionAgent:
    """Vision Language Model agent for game control."""
    
    def __init__(self, model_name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"):
        """
        Initialize the vision model.
        
        Args:
            model_name: HuggingFace model name/path
        """
        print(f"Loading model: {model_name}")
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        
        FastVisionModel.for_inference(self.model)
        
        goal = "Navigate to lantern."
        
        self.system_prompt = f"""You are an elite stealth agent playing 'The Dark Mod'.
        Your goal is to {goal}. You are in a 3D environment and must use the provided controls.

        CRITICAL RULES:
        1. Set "facing_target" to true ONLY if you can actually SEE the target in the current view
        2. If you CANNOT see the target (facing_target=false):
        - Use mouse_look to search (rotate camera)
        - Set action_sequence to [] (empty list - NO movement)
        - Stay in place and look around until you find it
        3. Once you SEE the target (facing_target=true):
        - Use mouse_look to keep it centered
        - Use action_sequence to move towards it (e.g. ["w", "w"])

        CONTROLS:
        - MOUSE: Look around (x: -100 to 100 for left/right, y: -50 to 50 for up/down)
        - 'w': Move Forward
        - 's': Move Backward  
        - 'a': Strafe Left
        - 'd': Strafe Right
        - 'c': Toggle Crouch (Use when Light Gem is bright/white)
        - 'space': Jump / Mantle
        - 'enter': Interact / Open Door / Pick up Loot

        HUD GUIDE:
        - Bottom Center Icon: Light Gem (Dark/Black = Hidden, Bright/White = Visible)
        - Red Bar: Health
        - Compass: Direction

        Respond with ONLY a valid JSON object in this exact format:
        {{
        "reasoning": "your detailed analysis",
        "visibility_status": "hidden" or "visible" or "partially_visible",
        "detected_threats": "description of any guards or threats",
        "facing_target": true or false,
        "mouse_look": {{"x": 100, "y": 40}},
        "action_sequence": ["w", "w", "w"],
        "action_explanation": "brief explanation",
        "goal_reached": true or false,
        "goal_status": "description of goal progress"
        }}

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
        Analyze a game frame and return structured action.
        
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
            min_p=min_p,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: str) -> GameAction:
        """Parse JSON response into GameAction."""
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return GameAction(**data)
            
            data = json.loads(cleaned_response)
            return GameAction(**data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse JSON output: {e}")
            print(f"Raw response: {response[:500]}...")
            return self._fallback_parsing(response)
    
    def _fallback_parsing(self, response: str) -> GameAction:
        """Fallback parsing when JSON parsing fails."""
        reasoning = response
        visibility_status = "hidden"
        detected_threats = "None detected"
        facing_target = False
        mouse_look = {"x": 0, "y": 0}
        action_sequence = ["none"]
        action_explanation = "Based on analysis"
        goal_reached = False
        goal_status = "In progress"
        
        action_match = re.search(r'[Aa]ction[:\s]+(\w)', response)
        if action_match:
            action = action_match.group(1).lower()
            if action in ['w', 's', 'a', 'd', 'c', 'space', 'enter']:
                action_sequence = [action]
        
        if any(word in response.lower() for word in ['bright', 'visible', 'light', 'white']):
            visibility_status = "visible"
        elif any(word in response.lower() for word in ['partially', 'somewhat']):
            visibility_status = "partially_visible"
        
        if any(word in response.lower() for word in ['guard', 'enemy', 'patrol', 'threat']):
            detected_threats = "Guard or threat detected"
        
        if any(word in response.lower() for word in ['facing', 'looking at', 'see the', 'visible']):
            facing_target = True
        
        if any(word in response.lower() for word in ['goal reached', 'objective complete', 'reached', 'arrived']):
            goal_reached = True
            goal_status = "Goal reached"
            action_sequence = []
        
        return GameAction(
            reasoning=reasoning,
            visibility_status=visibility_status,
            detected_threats=detected_threats,
            facing_target=facing_target,
            mouse_look=mouse_look,
            action_sequence=action_sequence,
            action_explanation=action_explanation,
            goal_reached=goal_reached,
            goal_status=goal_status
        )
    
    def update_goal(self, new_goal: str):
        """Update the agent's goal dynamically."""
        self.system_prompt = self.system_prompt.split("Your goal is to")[0] + \
                           f"Your goal is to {new_goal}. Remember you are in 3D environment" + \
                           self.system_prompt.split("and must use the provided controls")[1]


def main():
    """Example usage."""
    agent = VisionAgent()
    
    image_path = "imgs/example_darkmod.png"
    image = Image.open(image_path)
    
    print(f"\nAnalyzing image: {image_path}")
    print("=" * 80)
    
    action = agent.analyze_frame(image)
    
    print("\nSTRUCTURED OUTPUT:")
    print(f"Reasoning: {action.reasoning}")
    print(f"Visibility: {action.visibility_status}")
    print(f"Threats: {action.detected_threats}")
    print(f"Facing Target: {action.facing_target}")
    print(f"Mouse Look: {action.mouse_look}")
    print(f"Actions: {action.action_sequence}")
    print(f"Explanation: {action.action_explanation}")
    print(f"Goal Status: {action.goal_status}")


if __name__ == "__main__":
    main()
