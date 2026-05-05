import os
import glob
import torch
from config import ModelConfig
torch.serialization.add_safe_globals([ModelConfig])

def save_checkpoint(model, optimizer, step, config_name, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': model.config, # Save the full config object
        'config_name': config_name
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def load_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, None, 0, None, None
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if not checkpoints:
        return None, None, 0, None, None
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    latest_ckpt = checkpoints[-1]
    
    print(f"Loading checkpoint {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location='cpu', weights_only=True)
    return (
        checkpoint['model_state_dict'], 
        checkpoint['optimizer_state_dict'], 
        checkpoint['step'], 
        checkpoint.get('config_name'),
        checkpoint.get('config')
    )

def cleanup_checkpoints(checkpoint_dir, keep_last=5):
    """Keep only the most recent N checkpoints."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "step_*.pt"))
    if len(checkpoints) <= keep_last:
        return
    
    # Sort by step number
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    # Remove older ones
    for ckpt in checkpoints[:-keep_last]:
        try:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {ckpt}")
        except Exception as e:
            print(f"Error removing {ckpt}: {e}")
