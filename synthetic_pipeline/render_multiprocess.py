#!/usr/bin/env python3
"""
Multi-process rendering script for HOI sequences.
Each process handles one obj_id with a dedicated GPU.

Multiprocess-safe logging:
- Each child process has its own logger and log file
- Main process logs to console and main.log
- Child processes only log to their individual files to avoid stdout conflicts
"""

import logging
import rootutils
__ROOT__ = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
import glob
import subprocess
import time
from datetime import datetime

from graspxl_dataloader import GraspXLLoader
import signal

mp.set_start_method("spawn", force=True) # this is important! otherwise PyTorch/torch/cuda/open3d  will crash!!

def setup_logging(log_dir: str, process_id: int):
    """Setup logging for each process - multiprocess safe"""
    # Simple filename without timestamp
    log_file = os.path.join(log_dir, f"render_process_{process_id}.log")
    
    # Create a process-specific logger to avoid conflicts
    logger = logging.getLogger(f'process_{process_id}')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    
    # Clear any existing handlers to avoid duplication
    logger.handlers.clear()
    
    # File handler for this process
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' mode to overwrite
    file_handler.setLevel(logging.DEBUG)  # Capture DEBUG and above (includes WARNING, ERROR)
    file_formatter = logging.Formatter(
        '%(asctime)s - Process %(process)d - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to parent loggers to avoid conflicts
    logger.propagate = False
    
    # Force immediate flush
    file_handler.flush()
    
    # Optional: Add console handler for debugging (uncomment if needed)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    # console_handler.setFormatter(file_formatter)
    # logger.addHandler(console_handler)
    
    return logger

def run_blender(blender_executable, tmp_mesh_dir, render_out_dir, gpu_id, verbose=False, log_prefix=None):
    # blender_executable = f"{__ROOT__}/render_script/blender-3.2.2-linux-x64/blender"
    blender_script = f"{__ROOT__}/render_script/blender_seq_script.py"
    
    os.makedirs(render_out_dir, exist_ok=True)

    blender_cmd = [
        "xvfb-run", "-a",
        blender_executable, "-b",
        "--python", blender_script, "--",
        "--object_dir", tmp_mesh_dir,
        "--output_dir", render_out_dir
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id >= 0 else ""

    if verbose:
        print("Running Blender rendering command:")
        print(" ".join(blender_cmd))
        # Run and capture output
        result = subprocess.run(blender_cmd, env=env, check=False,
                            capture_output=True, text=True)
        
        # Save logs
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_prefix = f"/data1/DATA/vggt-logs/blender" if log_prefix is None else log_prefix
        os.makedirs(log_prefix, exist_ok=True)
        
        # Save combined log with metadata
        with open(f"{log_prefix}/{time_stamp}.log", 'w') as f:
            f.write(f"Command: {' '.join(blender_cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"Object dir: {tmp_mesh_dir}\n")
            f.write(f"Output dir: {render_out_dir}\n")
            f.write(f"GPU ID: {gpu_id}\n")
            f.write("-" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n" + "-" * 80 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr)
    
        
    else:
        result = subprocess.run(blender_cmd, env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Only print if there's an error
    if result.returncode != 0:
        print(f"Blender error: {result.stderr}")

    
    # After Blender rendering, process depth visualizations
    # print("\nProcessing depth visualizations...")
    # process_depth_visualizations(render_out_dir)


def render_single_obj(args_tuple):
    """
    Render all sequences for a single obj_id.
    This function runs in a separate process.
    
    Args:
        args_tuple: (obj_id, gpu_id, args) where args contains all configuration
    """
    obj_id, gpu_id, args = args_tuple
    
    # Setup logging for this process
    logger = setup_logging(args.log_dir, os.getpid())
    
    # Log process startup information  
    logger.info(f"=== Process started for obj_id: {obj_id} ===")
    logger.info(f"Assigned GPU ID: {gpu_id}")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Base directory: {args.base_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Override GPU ID for this process
        args.gpu_id = gpu_id
        logger.info(f"Args loaded. objaverse_dir={args.objaverse_dir}")
        logger.info(f"mano_dir={args.mano_dir}, dart_dir={args.dart_dir}, gpu_id={args.gpu_id}")
        
        logger.info("Initializing GraspXLRender...")
        logger.info("Step 1: Creating GraspXLRender object...")
        
        # Try to catch specific initialization steps
        try:
            renderer = GraspXLLoader(args)
            logger.info("GraspXLRender initialized successfully")
            logger.info(f"Renderer using PyTorch device: {renderer.device}")
        except Exception as init_e:
            logger.error(f"GraspXLRender initialization failed: {str(init_e)}")
            logger.error(f"Init exception type: {type(init_e).__name__}")
            import traceback
            logger.error(f"Init traceback: {traceback.format_exc()}")
            raise init_e
        
        # Find all .npy files for this obj_id
        obj_dir = os.path.join(args.base_dir, obj_id)
        if not os.path.exists(obj_dir):
            logger.warning(f"Object directory not found: {obj_dir}")
            return f"FAILED: {obj_id} - directory not found"
        
        npy_files = glob.glob(os.path.join(obj_dir, "*.npy"))
        if not npy_files:
            logger.warning(f"No .npy files found for obj_id: {obj_id}")
            return f"FAILED: {obj_id} - no sequences found"
        
        npy_files.sort()
        logger.info(f"Processing {len(npy_files)} sequences for obj_id: {obj_id}")
        
        # Get motion_id_list from args (already parsed in main)
        motion_id_list = args.motion_id_list
        success_count = 0
        logger.info(f"Starting to process {len(npy_files)} files...")
        
        for i, motion_path in enumerate(npy_files):
            # Skip motions not in motion_id_list if specified
            npy_id = motion_path.split("/")[-1].split(".")[0]  # e.g., "1" or "mano_1"
            # Extract numeric ID from both "1" and "mano_1" patterns
            if npy_id.startswith("mano_"):
                numeric_id = int(npy_id.split("_")[1])
            else:
                numeric_id = int(npy_id)

            if motion_id_list is not None and numeric_id not in motion_id_list:
                continue
            
            logger.info(f"Processing file {i+1}/{len(npy_files)}: {motion_path}")
            
            try:
                motion_id = Path(motion_path).stem
                logger.info(f"Motion ID: {motion_id}")
                
                tmp_output_dir = f"{args.tmp_prefix}/{obj_id}/{motion_id}"
                render_output_dir = os.path.join(args.output_dir, obj_id, motion_id)
                
                logger.info(f"Temp output dir: {tmp_output_dir}")
                logger.info(f"Render output dir: {render_output_dir}")
                
                # Export HOI sequence
                logger.info(f"Starting export_hoi_sequence for {motion_path}")
                renderer.export_hoi_sequence(motion_path, tmp_output_dir)
                logger.info(f"Completed export_hoi_sequence for {motion_path}")
                
                logger.info(f"Starting Blender rendering...")
                run_blender(args.blender_path, tmp_output_dir, render_output_dir, args.gpu_id, args.verbose, os.path.join(args.log_dir, "blender_logging"))
                logger.info(f"Completed Blender rendering")
                
                # Export additional info if rendering succeeded
                if os.path.exists(render_output_dir):
                    logger.info(f"Export info for {motion_path}")
                    renderer.export_info(motion_path, render_output_dir, tmp_output_dir)
                    success_count += 1
                    logger.info(f"Successfully completed {motion_path}")
                else:
                    logger.warning(f"Render output directory not found: {render_output_dir}")
                
                # Clean up temporary files
                if os.path.exists(f"{args.tmp_prefix}/{obj_id}"):
                    import shutil
                    shutil.rmtree(f"{args.tmp_prefix}/{obj_id}")
                    logger.info(f"Cleaned up temp directory: {args.tmp_prefix}/{obj_id}")
                
                logger.info(f"=== Completed processing {motion_path} ===")
                
            except Exception as e:
                logger.error(f"Exception processing {motion_path}: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
            
            finally:
                # Force flush after each file to ensure logs are written immediately
                for handler in logger.handlers:
                    handler.flush()
        
        result = f"SUCCESS: {obj_id} - {success_count}/{len(npy_files)} sequences completed"
        logger.info(result)
        logger.info(f"=== Process completed for obj_id: {obj_id} ===")
        
        # Force flush all handlers
        for handler in logger.handlers:
            handler.flush()
        
        return result
        
    except Exception as e:
        error_msg = f"FAILED: {obj_id} - {str(e)}"
        logger.error(error_msg)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception args: {e.args}")
        logger.error(f"=== Process failed for obj_id: {obj_id} ===")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Force flush all handlers
        for handler in logger.handlers:
            handler.flush()
            
        return error_msg


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-process HOI sequence rendering")
    
    # Essential paths
    parser.add_argument('--obj_type', type=str, default="small",
                       help='Object type (small/medium/large)')
    parser.add_argument('--blender_path', type=str, 
                        default=f"{__ROOT__}/render_script/blender-3.2.2-linux-x64/blender",
                       help='Path to the Blender executable.')
    parser.add_argument('--base_dir', type=str, 
                       default="/data1/DATA/GraspXL/mano_dataset_1",
                       help='Base directory path (obj_type will be appended)')
    parser.add_argument('--output_dir', type=str,
                       default="/data1/DATA/Objaverse-HOIData-mano",
                       help='Output directory path (obj_type will be appended)')
    
    # Resource allocation
    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                       help='Comma-separated GPU IDs to use (e.g., "0,1,2,3")')
    parser.add_argument('--max_processes', type=int, default=None,
                       help='Maximum number of parallel processes (default: number of GPUs)')
    
    # Object selection
    parser.add_argument('--obj_ids', type=str, default=None,
                       help='Comma-separated obj_ids to process (default: all)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start index in obj_id list')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End index in obj_id list (exclusive)')
    
    # Model paths (use render.py defaults)
    parser.add_argument('--objaverse_dir', type=str, 
                       default="/data1/DATA/graspxl-objaverse")
    parser.add_argument('--graspxl_dir', type=str, 
                       default="/data1/DATA/GraspXL")
    parser.add_argument('--mano_dir', type=str, 
                       default=f"{__ROOT__}/assets/mano_v1_2")
    parser.add_argument('--handy_dir', type=str,
                       default=f"{__ROOT__}/assets/handy")
    parser.add_argument('--dart_dir', type=str, default="/data1/DATA/DARTset",
                        help='Directory to the DARTset data.')
    
    # MANO beta prior
    parser.add_argument('--hamer_prior_path', type=str, 
                       default='/data1/DATA/HaMeR_DATA/_DATA/data/mano_mean_params.npz',
                       help='Path to HaMeR MANO mean parameters for beta prior.')
    parser.add_argument('--mano_beta_std', type=float, default=0.2,
                       help='Standard deviation for sampling MANO beta parameters. Applied per dimension.')
    
    # Temporary and logging
    parser.add_argument('--tmp_prefix', type=str, default=None,
                       help='Temporary mesh directory (default: <output_dir>/tmp_mesh)')
    parser.add_argument('--log_dir', type=str,
                       default="/data1/DATA/vggt-logs/render_multiprocess")
    parser.add_argument('--verbose', action='store_true', help='Whether to print blender verbose output.')

    # select model for hand (handy or mano)
    parser.add_argument('--use_handy', action='store_true', help='Whether to use HANDY model.')
    
    # Motion filtering
    parser.add_argument('--motion_ids', type=str, default=None,
                       help='Comma-separated motion indices to render (e.g., "0" or "0,1,2"). Default: render all motions.')
    
    return parser.parse_args()


def main():
    args = parse_args()

    # Set default tmp_prefix if not specified
    if args.tmp_prefix is None:
        args.tmp_prefix = os.path.join(args.output_dir, "tmp_mesh")
    
    
    # Construct full paths by appending obj_type to base directories
    args.base_dir = os.path.join(args.base_dir, args.obj_type)
    args.output_dir = os.path.join(args.output_dir, args.obj_type)
    
    # Parse GPU list
    gpu_list = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    max_processes = args.max_processes or len(gpu_list)
    
    # Setup main logging (multiprocess safe)
    # Create date-time based log directory (accurate to minute)
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    dated_log_dir = os.path.join(args.log_dir, date_str)
    os.makedirs(dated_log_dir, exist_ok=True)
    
    # Use root logger for main process, but configure it carefully
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid conflicts
    root_logger.handlers.clear()
    
    # Main process file handler
    main_file_handler = logging.FileHandler(os.path.join(dated_log_dir, "main.log"))
    main_file_handler.setLevel(logging.DEBUG)
    main_formatter = logging.Formatter('%(asctime)s - MAIN - %(levelname)s - %(message)s')
    main_file_handler.setFormatter(main_formatter)
    root_logger.addHandler(main_file_handler)
    
    # Console handler (only for main process)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(main_formatter)
    root_logger.addHandler(console_handler)
    
    # Get obj_id list
    logging.info(f"Getting object ID list from {args.base_dir}")
    if args.obj_ids:
        obj_id_list = [obj_id.strip() for obj_id in args.obj_ids.split(',')]
        logging.info(f"Using specified obj_ids: {obj_id_list}")
    else:
        all_dirs = os.listdir(args.base_dir)
        obj_id_list = [f for f in all_dirs 
                       if os.path.isdir(os.path.join(args.base_dir, f))]
        obj_id_list.sort()
        logging.info(f"Found {len(obj_id_list)} object directories in {args.base_dir}")
    
    # Parse motion_ids to list of integers (do this once in main, not in each process)
    if args.motion_ids is not None:
        args.motion_id_list = [int(x.strip()) for x in args.motion_ids.split(',')]
        logging.info(f"Filtering motions by indices: {args.motion_id_list}")
    else:
        args.motion_id_list = None
        
    # Apply start/end indices
    end_idx = args.end_idx or len(obj_id_list)
    original_count = len(obj_id_list)
    obj_id_list = obj_id_list[args.start_idx:end_idx]
    logging.info(f"After applying indices [{args.start_idx}:{end_idx}]: {len(obj_id_list)} objects")
    
    if len(obj_id_list) == 0:
        logging.error(f"No object IDs found after filtering! Original count: {original_count}, start_idx: {args.start_idx}, end_idx: {end_idx}")
        return
    
    logging.info(f"Processing {len(obj_id_list)} objects with {max_processes} processes")
    logging.info(f"Object type: {args.obj_type}")
    logging.info(f"Base directory: {args.base_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Temp directory: {args.tmp_prefix}")
    logging.info(f"Available GPUs: {gpu_list}")
    logging.info(f"Object range: {args.start_idx} to {end_idx-1}")
    logging.info(f"Log directory: {dated_log_dir}")
    
    # Create process arguments
    logging.info("Creating process arguments...")
    process_args = []
    
    # Pre-assign GPUs randomly to balance load across GPUs from the start
    # This prevents all initial tasks from being assigned to the same GPU pattern
    import random
    gpu_assignments = [gpu_list[i % len(gpu_list)] for i in range(len(obj_id_list))]
    random.shuffle(gpu_assignments)  # Randomize to balance GPU usage
    
    # Update log_dir to use the dated directory
    args.log_dir = dated_log_dir
    
    for i, obj_id in enumerate(obj_id_list):
        gpu_id = gpu_assignments[i]
        process_args.append((obj_id, gpu_id, args))
        if i < 20:  # Log first 20 to verify randomization
            logging.info(f"Process {i+1}: obj_id={obj_id}, gpu_id={gpu_id}")
    
    # Log GPU distribution
    from collections import Counter
    gpu_counts = Counter(gpu_assignments)
    logging.info(f"GPU distribution: {dict(gpu_counts)}")
    
    logging.info(f"Created {len(process_args)} process arguments")
    logging.info("Starting multiprocess pool...")
    
    # Run multiprocessing
    start_time = time.time()
    
    try:
        with mp.Pool(processes=max_processes) as pool:
            logging.info(f"Pool created with {max_processes} processes")
            results = pool.map(render_single_obj, process_args)
            logging.info(f"Pool.map completed, got {len(results)} results")
    except Exception as e:
        logging.error(f"Multiprocessing failed: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return
    
    # Summary
    elapsed_time = time.time() - start_time
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    
    logging.info(f"\n{'='*60}")
    logging.info(f"RENDERING COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Total objects: {len(obj_id_list)}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {len(obj_id_list) - success_count}")
    logging.info(f"Total time: {elapsed_time:.2f}s")
    logging.info(f"Average time per object: {elapsed_time/len(obj_id_list):.2f}s")
    
    # Print detailed results
    logging.info("\nDetailed Results:")
    for result in results:
        if result.startswith("FAILED"):
            logging.warning(result)
        else:
            logging.info(result)


if __name__ == "__main__":
    main()
