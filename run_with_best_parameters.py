#!/usr/bin/env python3
"""
run_with_best_parameters.py
Python replacement for run_with_best_parameters.sh
Runs UASTHN evaluation with best parameters from the paper.
Works on Windows, Linux, and macOS without Slurm dependency.
"""

import subprocess
import sys
import os
from pathlib import Path


def submit_job_crop_ensemble(dc, st, shift, arch, en, com, model, method):
    """
    Execute evaluation with the specified parameters.
    Replaces the original submit_job_crop_ensemble bash function.
    """
    
    # Select evaluation script based on METHOD
    # In the bash script, METHOD selects different .sbatch files,
    # but they all run the same myevaluate.py with different configurations.
    # We'll map METHOD to the appropriate settings.
    
    if method == 1:
        print(f"\n[INFO] Running METHOD=1: STHN Validation")
        eval_type = "sthn_val"
    elif method == 2:
        print(f"\n[INFO] Running METHOD=2: STHN Standard")
        eval_type = "sthn"
    elif method == 3:
        print(f"\n[INFO] Running METHOD=3: IHN")
        eval_type = "ihn"
    else:
        print(f"[ERROR] Unknown METHOD={method}. Exiting.")
        sys.exit(1)
    
    # Parameters from the original bash script's sbatch --export line
    USEED = 0
    STDM = "all"
    AGG = "zero"
    CS = -1
    
    # Number of crop samples (from SLURM_ARRAY_TASK_ID logic)
    # Original: CROP=5 when array task ID <= 5
    CROP = 5
    
    # Batch size depends on CROP
    if CROP <= 5:
        BAN = 8
    else:
        BAN = 4
    
    # Build the Python command
    # This mirrors the exact command from eval_local_sparse_extended_2_ue1_ce_sthn.sbatch
    cmd = [
        "python3" if sys.platform != "win32" else "python",
        "./local_pipeline/myevaluate.py",
        "--dataset_name", "satellite_0_thermalmapping_135_train",
        "--eval_model", f"logs/local_he/{model}",
        "--val_positive_dist_threshold", str(dc),
        "--lev0",
        "--database_size", "1536",
        "--corr_level", "4",
        "--two_stages",
        "--fine_padding", "32",
        "--first_stage_ue",
        "--test",
        "--batch_size", str(BAN),
        "--ue_aug_method", "shift",
        "--ue_num_crops", str(CROP),
        "--ue_shift_crops_types", st,
        "--ue_shift", str(shift),
        "--ue_seed", str(USEED),
        "--ue_std_method", STDM,
        "--ue_agg", AGG,
        "--check_step", str(CS),
        "--ue_method", "augment_ensemble",
        "--ue_ensemble_load_models", en,
        "--arch", arch,
        "--ue_combine", com
    ]
    
    # Print the command for debugging
    print(f"[CMD] {' '.join(cmd)}\n")
    
    # Execute the evaluation
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            cwd=os.getcwd()  # Run from current directory
        )
        print(f"[SUCCESS] Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed with exit code {e.returncode}")
        print(f"[ERROR] Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"[ERROR] Could not find myevaluate.py at ./local_pipeline/myevaluate.py")
        print(f"[ERROR] Make sure you are running this script from the UASTHN project root directory.")
        return False


def main():
    """
    Main execution - replicates the final lines of run_with_best_parameters.sh
    """
    print("=" * 60)
    print("UASTHN Evaluation with Best Parameters")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("./local_pipeline/myevaluate.py").exists():
        print("[ERROR] myevaluate.py not found!")
        print("[ERROR] Please run this script from the UASTHN project root directory.")
        print("[ERROR] Current directory:", os.getcwd())
        sys.exit(1)
    
    # Best parameters from the paper (as configured in the bash script)
    MODEL = ("satellite_0_thermalmapping_135_nocontrast_dense_exclusion_"
             "largest_ori_train-2024-06-01_14-57-59-16ed57bd-e7c3-4575-"
             "8421-d180efbbff36/UAGL.pth")
    ST = "random"                          # Sampling type
    SHIFT = 32                             # Crop offset (pixels)
    ARCH = "IHN"                           # Architecture
    EN = "./local_pipeline/ensembles/ensemble_512_STHN.txt"  # Ensemble file
    COM = "max"                            # Merge function
    DC = 512                               # Distance of centers (meters)
    METHOD = 2                             # Evaluation method
    
    print(f"\nConfiguration:")
    print(f"  Model:       {MODEL}")
    print(f"  Architecture: {ARCH}")
    print(f"  DC (search):  {DC}m")
    print(f"  Crop Shift:   {SHIFT}px")
    print(f"  Crop Type:    {ST}")
    print(f"  N crops:      5")
    print(f"  Ensemble:     {EN}")
    print(f"  Combine:      {COM}")
    print(f"  Method:       {METHOD}")
    print(f"  Batch Size:   8")
    
    # Run evaluation
    success = submit_job_crop_ensemble(
        dc=DC,
        st=ST,
        shift=SHIFT,
        arch=ARCH,
        en=EN,
        com=COM,
        model=MODEL,
        method=METHOD
    )
    
    if success:
        print(f"\n[COMPLETE] UASTHN evaluation finished.")
        sys.exit(0)
    else:
        print(f"\n[FAILED] UASTHN evaluation encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()