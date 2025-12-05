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
import wandb
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer, TrainingArguments
from trl import SFTTrainer


def load_training_data(dataset_dir: str):
    """
    Load labeled segments with actual PIL images.
    
    Args:
        dataset_dir: Path to labeled_segments directory
        
    Returns:
        List of training samples with loaded images
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
            
            # Get image path and LOAD the actual image
            img_filename = messages[1]["content"][0]["image"]
            img_path = os.path.join(segment_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            image = Image.open(img_path).convert("RGB")
            
            # Get goal instruction from user message
            goal_text = messages[1]["content"][1]["text"]
            goal = goal_text.split("Goal: ")[1].split("\n")[0]
            
            # Get action sequence from assistant message
            assistant_content = json.loads(messages[2]["content"])
            action_sequence = assistant_content["action_sequence"]
            
            # Create training sample with loaded image
            sample = {
                "image": image,  # Actual PIL Image, not path!
                "goal": goal,
                "actions": action_sequence
            }
            
            all_samples.append(sample)
    
    print(f"✓ Loaded {len(all_samples)} training samples from {len(segment_dirs)} segments")
    return all_samples





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
    
    # Initialize Weights & Biases
    wandb.init(
        project="darkmod-gameplay-agent",
        name=f"qwen3vl-lora-r{LORA_R}-bs{BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}",
        config={
            "model": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "max_seq_length": MAX_SEQ_LENGTH,
        }
    )
    
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
    
    # Split into train/val
    print("\n3. Splitting data...")
    val_size = int(len(training_samples) * 0.1)  # 10% validation
    val_samples = training_samples[:val_size]
    train_samples = training_samples[val_size:]
    print(f"   • Training samples: {len(train_samples)}")
    print(f"   • Validation samples: {len(val_samples)}")
    
    # Format dataset for Unsloth (list of dicts with "messages" key)
    def convert_to_conversation(sample):
        """Convert a sample to Unsloth conversation format."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},  # Include image directly
                    {"type": "text", "text": f"Goal: {sample['goal']}\n\nPredict the next 7 actions:"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps({"actions": sample["actions"]})}
                ]
            }
        ]
        return {"messages": conversation}
    
    print("\n4. Converting to Unsloth format...")
    train_dataset = [convert_to_conversation(s) for s in train_samples]
    val_dataset = [convert_to_conversation(s) for s in val_samples]
    print(f"✓ Converted datasets")
    
    # Training configuration (using TrainingArguments for TRL 0.24.0)
    print("\n5. Configuring trainer...")
    
    FastVisionModel.for_training(model)  # Enable for training!
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="",  # Required for vision
        dataset_kwargs={"skip_prepare_dataset": True},  # Required for vision
        max_seq_length=MAX_SEQ_LENGTH,  # Pass to SFTTrainer, not TrainingArguments
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=10,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            
            # Optimization
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            
            # Memory optimization
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            
            # Logging and saving
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            
            # Output
            output_dir=OUTPUT_DIR,
            report_to="wandb",
            seed=3407,
            run_name=f"qwen3vl-lora-r{LORA_R}-bs{BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}",
            
            # CRITICAL: You MUST put this for vision finetuning:
            remove_unused_columns=False,
        ),
    )
    
    # Train
    print("\n6. Starting training...")
    print(f"   • Training samples: {len(train_dataset)}")
    print(f"   • Validation samples: {len(val_dataset)}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   • Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   • Epochs: {NUM_EPOCHS}")
    print(f"   • Learning rate: {LEARNING_RATE}")
    print(f"   • LoRA r={LORA_R}, alpha={LORA_ALPHA}")
    print()
    
    trainer_stats = trainer.train()
    
    # Save final model
    print("\n7. Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Final training loss: {trainer_stats.training_loss:.4f}")
    if hasattr(trainer_stats, 'metrics') and 'eval_loss' in trainer_stats.metrics:
        print(f"Final validation loss: {trainer_stats.metrics['eval_loss']:.4f}")
    print("="*80)
    
    # Test inference
    print("\n8. Testing inference...")
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
    
    # Finish wandb run
    wandb.finish()
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
