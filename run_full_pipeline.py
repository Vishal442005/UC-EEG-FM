"""
Master Turnkey Pipeline Runner for UC-EEG-FM.
Provides unified command-line execution for end-to-end dataset downloading,
preprocessing, pre-training, clinical benchmarking, interpretability, and integrity auditing.
"""

import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name, description):
    script_path = os.path.join(BASE_DIR, script_name)
    print("\n" + "=" * 80)
    print(f"RUNNING: {description} ({script_name})")
    print("=" * 80 + "\n", flush=True)
    res = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"\n[ERROR] Pipeline step failed: {script_name} exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n[SUCCESS] Completed: {description}")

def main():
    parser = argparse.ArgumentParser(
        description="UC-EEG-FM: Unified Cross-Disorder EEG Foundation Model Turnkey Reproduction Suite"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "download", "phase1", "phase2", "phase3", "phase4_5", "phase6", "audit"],
        default="audit",
        help="Pipeline phase to execute. Default: 'audit' (verifies all artifacts and benchmarks)."
    )
    
    args = parser.parse_args()
    
    print("*" * 80)
    print("UNIFIED CROSS-DISORDER EEG FOUNDATION MODEL (UC-EEG-FM)")
    print(f"Executing Phase: {args.phase.upper()}")
    print("*" * 80)
    
    if args.phase in ["all", "download"]:
        run_script("download_genuine_datasets.py", "Anonymous AWS S3 Clinical Dataset Streaming")
        
    if args.phase in ["all", "phase1"]:
        run_script("create_master_metadata.py", "Cohort Metadata Harmonization (146 Subjects)")
        run_script("run_phase1_feature_extraction.py", "Standardized FIR Filtering & 11 Biomarker Extraction")
        
    if args.phase in ["all", "phase2"]:
        run_script("train_phase2_foundation_model.py", "Self-Supervised Spatial-Temporal Transformer Pre-training")
        run_script("plot_phase2_latent_manifold.py", "Latent Manifold Topology & PCA/t-SNE Projection")
        
    if args.phase in ["all", "phase3"]:
        run_script("run_phase3_clinical_benchmarks.py", "Strict Subject-Stratified Group 5-Fold Benchmarking")
        
    if args.phase in ["all", "phase4_5"]:
        run_script("run_phase4_transfer_and_saliency.py", "Few-Shot Sample Efficiency & Neurobiological Saliency")
        run_script("run_expanded_phases_3_4_5.py", "Multi-Model Benchmarks, Bidirectional Transfer & Topomaps")
        
    if args.phase in ["all", "phase6"]:
        run_script("run_phase6_clinical_decision_support.py", "Real-Time CDSS, Edge Profiling & Patient Dossiers")
        
    if args.phase in ["all", "audit"]:
        run_script("run_master_integrity_audit.py", "Master Integrity & Consistency Audit")

    print("\n" + "*" * 80)
    print("PIPELINE EXECUTION COMPLETE: ALL REQUESTED TASKS FINISHED SUCCESSFULLY")
    print("*" * 80)

if __name__ == "__main__":
    main()
