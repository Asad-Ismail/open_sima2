"""
High-level labeling script for video segments using Qwen VLM.
Labels 2-3 second video clips with language instructions and frame-by-frame actions.
Creates Unsloth-ready training data for Qwen3-VL fine-tuning.
"""

import os
import json
import glob
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from unsloth import FastVisionModel


class VideoSegmentLabeler:
    """Labels video segments with high-level goals and frame-by-frame actions."""
    
    def __init__(self, 
                 model_name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
                 fps: int = 10,
                 segment_duration: float = 4.0):
        """
        Initialize the labeler.
        
        Args:
            model_name: Qwen VLM model name
            fps: Frames per second of recorded video
            segment_duration: Duration of each video segment in seconds
        """
        print(f"Loading Qwen VLM: {model_name}")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )
        FastVisionModel.for_inference(self.model)
        
        self.fps = fps
        self.segment_duration = segment_duration
        self.frames_per_segment = int(fps * segment_duration)  # ~40 frames for 4s at 10fps
        
        # Instruction extraction prompt - extracts the high-level goal
        self.instruction_prompt = """Look at these game frames from 'The Dark Mod' showing the START and END of a player's action.

What was the player's GOAL? Answer with a SHORT instruction (3-5 words) that describes their objective.

Good examples:
- "Go to the lantern"
- "Navigate to the door"
- "Check the posters"
- "Climb the stairs"
- "Search the room"
- "Approach the guard"

Bad examples (too detailed):
- "Turn right 90 degrees and move forward to reach the lantern"
- "The player slowly turns their camera to the left while crouching"

Answer with ONLY the short goal instruction, nothing else."""

    def extract_goal_instruction(self, frames: List[Image.Image]) -> str:
        """
        Extract the high-level goal/instruction from the sequence using VLM.
        
        Args:
            frames: List of PIL Images (full segment)
            
        Returns:
            Short goal-oriented instruction (3-5 words)
        """
        # Sample key frames: start, middle, end
        sample_indices = [0, len(frames)//2, len(frames)-1]
        sampled_frames = [frames[i] for i in sample_indices if i < len(frames)]
        
        # Use first and last frame to show what changed
        first_frame = sampled_frames[0]
        last_frame = sampled_frames[-1]
        
        # Create a side-by-side comparison or pass both frames
        # For now using first frame, but ideally pass both to show progression
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.instruction_prompt}
            ]}
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.tokenizer(
            first_frame,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.3,
            use_cache=True,
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        
        if response.startswith(prompt_text):
            response = response[len(prompt_text):].strip()
        
        # Clean up response and return
        goal = response.strip().strip('"').strip("'")
        return goal

    def load_segment_frames(self, 
                           dataset_dir: str, 
                           session_id: str,
                           start_idx: int) -> Tuple[List[Image.Image], List[Dict]]:
        """
        Load a segment of frames and their action labels.
        
        Args:
            dataset_dir: Path to dataset directory
            session_id: Session ID to filter files
            start_idx: Starting frame index
            
        Returns:
            Tuple of (frames, action_labels)
        """
        # Get all files for this session
        pattern = os.path.join(dataset_dir, f"{session_id}_*.jpg")
        all_files = sorted(glob.glob(pattern))
        
        if start_idx >= len(all_files):
            return [], []
        
        # Get segment files
        end_idx = min(start_idx + self.frames_per_segment, len(all_files))
        segment_files = all_files[start_idx:end_idx]
        
        frames = []
        action_labels = []
        
        for img_path in segment_files:
            # Load image
            frames.append(Image.open(img_path))
            
            # Load corresponding action label
            json_path = img_path.replace('.jpg', '.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    action_labels.append(json.load(f))
            else:
                action_labels.append({"action_sequence": ["none"]})
        
        return frames, action_labels

    def create_sliding_window_labels(self,
                                     frames: List[Image.Image],
                                     action_labels: List[Dict],
                                     instruction: str,
                                     window_size: int = 7) -> List[Dict]:
        """
        Create training samples with sliding window of future actions.
        
        Args:
            frames: List of frames in segment
            action_labels: Original action labels for each frame
            instruction: Short goal-oriented instruction for this segment
            window_size: Number of future actions to predict (default 7)
            
        Returns:
            List of training samples
        """
        training_samples = []
        
        for i, frame in enumerate(frames):
            # Collect next 'window_size' actions
            future_actions = []
            
            for j in range(window_size):
                future_idx = i + j
                if future_idx < len(action_labels):
                    # Get the action from this frame
                    action = action_labels[future_idx].get("action_sequence", ["none"])
                    # Take first action if multiple (or handle as needed)
                    future_actions.extend(action if action != ["none"] else ["none"])
                else:
                    # Past end of segment = task complete
                    future_actions.append("none")
            
            # Trim to exactly window_size actions
            future_actions = future_actions[:window_size]
            
            # Pad with 'none' if needed
            while len(future_actions) < window_size:
                future_actions.append("none")
            
            # Create training sample
            sample = {
                "image": frame,
                "instruction": instruction,
                "action_sequence": future_actions,
                "frame_idx": i,
                "segment_total_frames": len(frames)
            }
            
            training_samples.append(sample)
        
        return training_samples

    def save_unsloth_format(self,
                           training_samples: List[Dict],
                           output_dir: str,
                           segment_id: str):
        """
        Save training samples in Unsloth-compatible format.
        
        Args:
            training_samples: List of training samples
            output_dir: Output directory
            segment_id: Unique segment identifier
        """
        os.makedirs(output_dir, exist_ok=True)
        segment_dir = os.path.join(output_dir, segment_id)
        os.makedirs(segment_dir, exist_ok=True)
        
        # Save each frame with its label
        for idx, sample in enumerate(training_samples):
            # Save image
            img_path = os.path.join(segment_dir, f"frame_{idx:04d}.jpg")
            sample["image"].save(img_path)
            
            # Create Qwen3-VL training format
            training_data = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a game-playing agent. Given a goal instruction, predict the action sequence to accomplish it from the current frame."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"frame_{idx:04d}.jpg"},
                            {"type": "text", "text": f"Goal: {sample['instruction']}\n\nPredict the next 7 actions to execute."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "action_sequence": sample["action_sequence"],
                            "reasoning": f"To accomplish '{sample['instruction']}', the next actions are: {', '.join(sample['action_sequence'][:3])}..."
                        })
                    }
                ]
            }
            
            # Save label
            label_path = os.path.join(segment_dir, f"frame_{idx:04d}.json")
            with open(label_path, 'w') as f:
                json.dump(training_data, f, indent=2)
        
        print(f"  ✓ Saved {len(training_samples)} training samples to {segment_dir}")

    def process_session(self, 
                       dataset_dir: str,
                       output_dir: str,
                       session_id: str):
        """
        Process an entire recording session into labeled segments.
        
        Args:
            dataset_dir: Input dataset directory
            output_dir: Output directory for training data
            session_id: Session ID to process
        """
        print(f"\n{'='*80}")
        print(f"Processing session: {session_id}")
        print(f"{'='*80}")
        
        # Get all frames for this session
        pattern = os.path.join(dataset_dir, f"{session_id}_*.jpg")
        all_files = sorted(glob.glob(pattern))
        total_frames = len(all_files)
        
        print(f"Total frames: {total_frames}")
        print(f"Segment size: {self.frames_per_segment} frames (~{self.segment_duration}s)")
        
        segment_count = 0
        start_idx = 0
        
        while start_idx < total_frames:
            print(f"\n--- Segment {segment_count + 1} (frames {start_idx}-{start_idx + self.frames_per_segment}) ---")
            
            # Load segment frames
            frames, action_labels = self.load_segment_frames(
                dataset_dir, session_id, start_idx
            )
            
            if not frames:
                break
            
            print(f"  Loaded {len(frames)} frames")
            
            # Extract goal instruction
            print(f"  Extracting goal instruction with VLM...")
            instruction = self.extract_goal_instruction(frames)
            print(f"  Instruction: '{instruction}'")
            
            # Create sliding window labels
            print(f"  Creating sliding window labels...")
            training_samples = self.create_sliding_window_labels(
                frames, action_labels, instruction
            )
            
            # Save in Unsloth format
            segment_id = f"{session_id}_seg{segment_count:03d}"
            self.save_unsloth_format(training_samples, output_dir, segment_id)
            
            segment_count += 1
            start_idx += self.frames_per_segment
        
        print(f"\n✅ Processed {segment_count} segments from session {session_id}")


def main():
    """Main labeling pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Label video segments for VLM training")
    parser.add_argument("--input_dir", type=str, default="dataset/raw_playthrough",
                       help="Input directory with raw recordings")
    parser.add_argument("--output_dir", type=str, default="dataset/labeled_segments",
                       help="Output directory for labeled training data")
    parser.add_argument("--session_id", type=str, default=None,
                       help="Specific session ID to process (default: process all)")
    
    args = parser.parse_args()
    
    # Initialize labeler
    labeler = VideoSegmentLabeler()
    
    # Get all unique session IDs
    if args.session_id:
        session_ids = [args.session_id]
    else:
        # Extract session IDs from filenames
        pattern = os.path.join(args.input_dir, "*.jpg")
        all_files = glob.glob(pattern)
        session_ids = list(set([
            os.path.basename(f).split('_')[0] + "_" + os.path.basename(f).split('_')[1]
            for f in all_files
        ]))
        session_ids.sort()
    
    print(f"Found {len(session_ids)} sessions to process")
    
    # Process each session
    for session_id in session_ids:
        try:
            labeler.process_session(args.input_dir, args.output_dir, session_id)
        except Exception as e:
            print(f"❌ Error processing session {session_id}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("✅ Labeling complete!")
    print(f"Training data saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
