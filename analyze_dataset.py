"""
Analyze the labeled dataset before training.
Check action distribution, goal diversity, and data quality.
"""

import os
import json
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_dataset(dataset_dir: str = "dataset/labeled_segments"):
    """
    Comprehensive dataset analysis.
    """
    print("="*80)
    print("DATASET ANALYSIS")
    print("="*80)
    
    all_actions = []
    all_goals = []
    actions_per_frame = []
    segment_stats = []
    goal_action_map = defaultdict(list)
    
    segment_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*_seg*")))
    total_frames = 0
    
    print(f"\n📁 Found {len(segment_dirs)} segments")
    print(f"📊 Analyzing...")
    
    for seg_idx, segment_dir in enumerate(segment_dirs):
        json_files = sorted(glob.glob(os.path.join(segment_dir, "frame_*.json")))
        
        if not json_files:
            continue
            
        # Get goal from first frame
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        
        goal_text = data["messages"][1]["content"][1]["text"]
        goal = goal_text.split("Goal: ")[1].split("\n")[0]
        
        segment_actions = []
        
        for json_path in json_files:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Get actions
            assistant_content = json.loads(data["messages"][2]["content"])
            actions = assistant_content["action_sequence"]
            
            # Collect stats
            all_actions.extend(actions)
            segment_actions.extend(actions)
            actions_per_frame.append(len([a for a in actions if a != "none"]))
            
            total_frames += 1
        
        all_goals.append(goal)
        goal_action_map[goal].extend(segment_actions)
        segment_stats.append({
            "segment": os.path.basename(segment_dir),
            "goal": goal,
            "frames": len(json_files),
            "actions": segment_actions
        })
    
    print(f"\n✓ Analyzed {total_frames} frames from {len(segment_dirs)} segments")
    
    # === BASIC STATS ===
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(f"Total frames: {total_frames}")
    print(f"Total segments: {len(segment_dirs)}")
    print(f"Avg frames per segment: {total_frames / len(segment_dirs):.1f}")
    print(f"Unique goals: {len(set(all_goals))}")
    
    # === ACTION DISTRIBUTION ===
    print("\n" + "="*80)
    print("ACTION DISTRIBUTION")
    print("="*80)
    
    action_counts = Counter(all_actions)
    total_actions = len(all_actions)
    
    print(f"\nTotal action predictions: {total_actions}")
    print(f"\nAction frequencies:")
    for action, count in action_counts.most_common():
        percentage = (count / total_actions) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {action:8s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    # Check sparsity
    none_count = action_counts.get("none", 0)
    none_percentage = (none_count / total_actions) * 100
    
    print(f"\n🎯 Sparsity Analysis:")
    print(f"  'none' actions: {none_count}/{total_actions} ({none_percentage:.1f}%)")
    
    if none_percentage > 70:
        print("  ⚠️  WARNING: Very sparse! >70% 'none' actions")
        print("     → Model may struggle to learn meaningful actions")
    elif none_percentage > 50:
        print("  ⚠️  CAUTION: Quite sparse. 50-70% 'none' actions")
        print("     → Consider collecting more dynamic gameplay")
    else:
        print("  ✓ Good density! <50% 'none' actions")
    
    # Non-none actions per frame
    avg_real_actions = sum(actions_per_frame) / len(actions_per_frame)
    print(f"\n  Avg non-'none' actions per frame: {avg_real_actions:.2f}")
    
    # === GOAL DISTRIBUTION ===
    print("\n" + "="*80)
    print("GOAL DISTRIBUTION")
    print("="*80)
    
    goal_counts = Counter(all_goals)
    print(f"\nTotal goals: {len(goal_counts)}")
    print(f"\nGoal frequencies:")
    for goal, count in goal_counts.most_common(10):
        print(f"  {count:3d}x: {goal}")
    
    if len(goal_counts.most_common()) > 10:
        print(f"  ... and {len(goal_counts) - 10} more goals")
    
    # Check balance
    max_count = max(goal_counts.values())
    min_count = min(goal_counts.values())
    balance_ratio = max_count / min_count
    
    print(f"\n🎯 Balance Analysis:")
    print(f"  Most common goal: {max_count} segments")
    print(f"  Least common goal: {min_count} segments")
    print(f"  Balance ratio: {balance_ratio:.1f}x")
    
    if balance_ratio > 5:
        print("  ⚠️  WARNING: Highly imbalanced dataset")
        print("     → Model may overfit to common goals")
    elif balance_ratio > 3:
        print("  ⚠️  CAUTION: Somewhat imbalanced")
    else:
        print("  ✓ Well balanced!")
    
    # === ACTION PATTERNS PER GOAL ===
    print("\n" + "="*80)
    print("ACTION PATTERNS BY GOAL (Top 5 goals)")
    print("="*80)
    
    for goal, count in goal_counts.most_common(5):
        goal_actions = goal_action_map[goal]
        goal_action_counts = Counter(goal_actions)
        total_goal_actions = len(goal_actions)
        
        print(f"\n📌 {goal} ({count} segments, {total_goal_actions} actions):")
        
        # Top 5 actions for this goal
        for action, action_count in goal_action_counts.most_common(5):
            pct = (action_count / total_goal_actions) * 100
            print(f"   {action:8s}: {pct:5.1f}%")
    
    # === DIVERSITY CHECK ===
    print("\n" + "="*80)
    print("DIVERSITY ANALYSIS")
    print("="*80)
    
    # Unique action sequences (as tuples)
    unique_sequences = set()
    for stats in segment_stats:
        # Sample first 7 actions from segment
        seq = tuple(stats["actions"][:7])
        unique_sequences.add(seq)
    
    print(f"\nUnique action sequences (first 7 actions): {len(unique_sequences)}")
    print(f"Total segments: {len(segment_stats)}")
    diversity_ratio = len(unique_sequences) / len(segment_stats)
    print(f"Diversity ratio: {diversity_ratio:.2f}")
    
    if diversity_ratio < 0.3:
        print("⚠️  WARNING: Low diversity! Many repeated patterns")
        print("   → Model may memorize rather than learn")
    elif diversity_ratio < 0.5:
        print("⚠️  CAUTION: Moderate diversity")
    else:
        print("✓ Good diversity!")
    
    # === MOVEMENT vs CAMERA ===
    print("\n" + "="*80)
    print("MOVEMENT vs CAMERA CONTROL")
    print("="*80)
    
    movement_keys = {"w", "s", "a", "d"}
    camera_keys = {"up", "down", "left", "right"}
    other_keys = {"c", "space", "enter"}
    
    movement_count = sum(action_counts.get(k, 0) for k in movement_keys)
    camera_count = sum(action_counts.get(k, 0) for k in camera_keys)
    other_count = sum(action_counts.get(k, 0) for k in other_keys)
    real_actions = total_actions - none_count
    
    print(f"\nMovement (w/a/s/d): {movement_count} ({movement_count/real_actions*100:.1f}% of non-none)")
    print(f"Camera (arrows):    {camera_count} ({camera_count/real_actions*100:.1f}% of non-none)")
    print(f"Other (c/space/enter): {other_count} ({other_count/real_actions*100:.1f}% of non-none)")
    
    if camera_count == 0:
        print("\n❌ CRITICAL: No camera actions recorded!")
        print("   → Model cannot learn camera control")
    elif camera_count < movement_count * 0.2:
        print("\n⚠️  WARNING: Very few camera actions")
        print("   → Model may struggle with camera control")
    
    # === RECOMMENDATIONS ===
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    
    if none_percentage > 70:
        issues.append("High sparsity (>70% 'none' actions)")
    if balance_ratio > 5:
        issues.append("Highly imbalanced goal distribution")
    if diversity_ratio < 0.3:
        issues.append("Low action sequence diversity")
    if camera_count < movement_count * 0.2:
        issues.append("Insufficient camera control examples")
    if total_frames < 2000:
        issues.append(f"Small dataset ({total_frames} frames, recommend >5000)")
    
    if issues:
        print("\n⚠️  Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n📋 Suggested actions:")
        if none_percentage > 70:
            print("  • Record more dynamic gameplay with continuous movement")
        if balance_ratio > 5:
            print("  • Collect more data for underrepresented goals")
        if diversity_ratio < 0.3:
            print("  • Record different routes/approaches to same goals")
        if camera_count < movement_count * 0.2:
            print("  • Ensure camera control is properly recorded (arrow keys)")
        if total_frames < 2000:
            print("  • Collect at least 30-60 minutes more gameplay")
        
        print("\n🚦 Training recommendation: PROCEED WITH CAUTION")
        print("   Model will train but may have limited capabilities.")
    else:
        print("\n✅ Dataset looks good! Ready for training.")
    
    print("\n" + "="*80)
    
    return {
        "total_frames": total_frames,
        "total_segments": len(segment_dirs),
        "action_counts": action_counts,
        "goal_counts": goal_counts,
        "none_percentage": none_percentage,
        "balance_ratio": balance_ratio,
        "diversity_ratio": diversity_ratio
    }


if __name__ == "__main__":
    stats = analyze_dataset()
