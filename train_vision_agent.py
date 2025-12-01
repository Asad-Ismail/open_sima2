"""
Fine-tune Qwen3-VL-8B on gameplay data using Unsloth LoRA.
Trains the model to predict action sequences given goal instructions and game frames.
"""

import os
import json
import glob
from pathlib import Path
from PIL import Image
import torch
from unsloth import FastVisionModel, is_bf16_supported
from transformers import TextStreamer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def load_training_data(dataset_dir: str):
    """
    Load labeled segments into training format.
    
    Args:
        dataset_dir: Path to labeled_segments directory
        
    Returns:
        List of training samples
    """
    print(f"Loading training data from: {dataset_dir}")
    
    all_samples = []
    segment_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*_seg*")))
    
    for segment_dir in segment_dirs:
        json_files = sorted(glob.glob(os.path.join(segment_dir, "frame_*.json")))
        
        for json_path in json_files:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract components
            messages = data["messages"]
            
            # Get image path (relative to segment dir)
            img_filename = messages[1]["content"][0]["image"]
            img_path = os.path.join(segment_dir, img_filename)
            
            # Get goal instruction from user message
            goal_text = messages[1]["content"][1]["text"]
            goal = goal_text.split("Goal: ")[1].split("\n")[0]
            
            # Get action sequence from assistant message
            assistant_content = json.loads(messages[2]["content"])
            action_sequence = assistant_content["action_sequence"]
            
            # Create simplified training sample (no system role, no reasoning)
            sample = {
                "image": img_path,
                "goal": goal,
                "actions": action_sequence
            }
            
            all_samples.append(sample)
    
    print(f"✓ Loaded {len(all_samples)} training samples from {len(segment_dirs)} segments")
    return all_samples


def format_for_training(sample):
    """
    Convert sample to Qwen3-VL training format.
    Remove system role and reasoning - just goal + actions.
    """
    # Simple conversation: user asks for actions given goal, assistant responds with action list
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": f"Goal: {sample['goal']}\n\nPredict the next 7 actions:"}
            ]
        },
        {
            "role": "assistant",
            "content": json.dumps({"actions": sample["actions"]})
        }
    ]
    
    return {"messages": messages}


def main():
    """Main training pipeline."""
    
    # Configuration
    DATASET_DIR = "dataset/labeled_segments"
    MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    OUTPUT_DIR = "models/gameplay_agent_lora"
    
    # LoRA configuration
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    
    # Training configuration
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_SEQ_LENGTH = 2048
    
    print("="*80)
    print("Qwen3-VL Gameplay Agent Fine-tuning")
    print("="*80)
    
    # Load model with LoRA
    print("\n1. Loading base model with LoRA adapters...")
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Load training data
    print("\n2. Loading training data...")
    training_samples = load_training_data(DATASET_DIR)
    
    if len(training_samples) == 0:
        print("❌ No training data found! Run label_segments.py first.")
        return
    
    # Format for training
    print("\n3. Formatting data for training...")
    formatted_samples = [format_for_training(s) for s in training_samples]
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_samples)
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    # Training configuration
    print("\n4. Configuring trainer...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        
        # Training hyperparameters
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        
        # Optimization
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        
        # Memory optimization
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        
        # Logging
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        
        # Other
        seed=3407,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="messages",
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    print("\n5. Starting training...")
    print(f"   • Total samples: {len(dataset)}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   • Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   • Epochs: {NUM_EPOCHS}")
    print(f"   • Learning rate: {LEARNING_RATE}")
    print(f"   • LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print()
    
    trainer_stats = trainer.train()
    
    # Save final model
    print("\n6. Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training loss: {trainer_stats.training_loss:.4f}")
    print("="*80)
    
    # Test inference
    print("\n7. Testing inference...")
    FastVisionModel.for_inference(model)
    
    # Load a sample image
    test_sample = training_samples[0]
    test_image = Image.open(test_sample["image"])
    test_goal = test_sample["goal"]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"Goal: {test_goal}\n\nPredict the next 7 actions:"}
            ]
        }
    ]
    
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        test_image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    print(f"\nTest Goal: {test_goal}")
    print("Generated actions:")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.5,
    )


if __name__ == "__main__":
    main()
