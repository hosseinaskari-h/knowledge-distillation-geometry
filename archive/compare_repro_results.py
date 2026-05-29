import json
import os
import glob
import numpy as np

# Paper Claims from Table 8 (Updated with validated values)
PAPER_CLAIMS = {
    # Identity
    "gpt2+gpt2_direct_random": 1.0000,
    "gpt2+gpt2_direct_same": 1.0000,
    "gpt2+gpt2_direct_text": 1.0000,
    "gpt2+gpt2_blended_0.5_random": 1.0000,
    "gpt2+gpt2_blended_0.3_random": 1.0000,
    "gpt2+gpt2_blended_0.7_random": 1.0000,
    "gpt2+gpt2_decay_0.1_random": 1.0000,
    "gpt2+gpt2_decay_0.3_random": 1.0000,
    
    # Distillation Self
    "distilgpt2+distilgpt2_direct_random": 1.0000,
    "distilgpt2+distilgpt2_blended_0.5_random": 1.0000,
    "distilgpt2+distilgpt2_decay_0.1_random": 1.0000,

    # Proj Control
    "gpt2+gpt2_direct_random_force": 0.9987,
    "gpt2+gpt2_direct_text_force": 0.9984,

    # Encoder
    "bert-base-uncased+distilbert-base-uncased_direct_random": 0.3182,
    "bert-base-uncased+bert-base-uncased_direct_random": 0.5721,

    # Fine-tuning
    "gpt2-medium+microsoft/DialoGPT-medium_direct_random": 0.9485,
    "gpt2-medium+gpt2-medium_direct_random": 1.0000,

    # Cross-Scale
    "gpt2+gpt2-medium_direct_random_force": 0.1220,
    "gpt2+gpt2-medium_blended_0.5_random_force": 0.0069,
    "gpt2+gpt2-large_direct_random_force": 0.3765,
    "gpt2+gpt2-large_blended_0.5_random_force": 0.2323,
    "gpt2-medium+gpt2-large_direct_random_force": 0.2779,

    # Distillation Pair
    "gpt2+distilgpt2_direct_random": 0.9986,
    "gpt2+distilgpt2_blended_0.5_random": 0.9974,
    "gpt2+distilgpt2_decay_0.1_random": 0.9982,

    # Cross-Architecture
    "gpt2+EleutherAI/gpt-neo-1.3B_direct_random_force": -0.0237,
    "gpt2+EleutherAI/gpt-neo-1.3B_blended_0.5_random_force": -0.0326,
    "gpt2+EleutherAI/gpt-neo-1.3B_blended_0.5_text_force": -0.0326,
    "gpt2+EleutherAI/gpt-neo-1.3B_decay_0.1_text_force": -0.0261,
    "gpt2-medium+EleutherAI/gpt-neo-1.3B_direct_random_force": 0.0452,
}

def load_latest_repro_report(dir_path="data/reproduction"):
    files = glob.glob(os.path.join(dir_path, "final_reproduction_report_*.json"))
    if files:
        latest_file = max(files, key=os.path.getctime)
        print(f"Loading final report: {latest_file}")
        with open(latest_file, 'r') as f:
            return json.load(f)
            
    # Fallback: Aggregate progress files
    prog_files = glob.glob(os.path.join(dir_path, "reproduction_progress_*.json"))
    if not prog_files:
        print("No results found.")
        return None
        
    print(f"Aggregating {len(prog_files)} progress files...")
    results = []
    # Deduplicate by looking at config index? 
    # Filenames have timestamps, but no index.
    # We load all, then we need to extract the 'latest' entry from each.
    # But wait, progress file X might be from step 1, progress file Y from step 2.
    # We want the LATEST progress file, which contains the latest step... 
    # WAIT. My script logic:
    # save_results({'configs_completed': i+1, 'latest': summary}, ...)
    # This creates a NEW file for each step.
    # So we have 41 files. Each file contains ONE result (the valid one for that step).
    # So we should load ALL of them, extract 'latest', and that gives us the full set!
    
    for pf in prog_files:
        with open(pf, 'r') as f:
            try:
                data = json.load(f)
                if 'latest' in data:
                    results.append(data['latest'])
            except:
                pass
                
    return {'results': results}

def get_config_key(entry):
    c = entry['config']
    ma = entry['model_a']
    mb = entry['model_b']
    
    coupling = c['coupling']
    key = f"{ma}+{mb}_{coupling}"
    
    if coupling == 'blended':
        key += f"_{c.get('alpha')}"
    elif coupling == 'decay':
        key += f"_{c.get('decay')}"
        
    key += f"_{c['init']}"
    
    # Check force_proj. In run_experiments.py, it's boolean.
    if c.get('force_proj'):
        key += "_force"
        
    return key

def compare():
    data = load_latest_repro_report()
    if not data:
        return

    results = data.get('results', [])
    
    # Sort results for consistent output
    results.sort(key=lambda x: get_config_key(x))

    with open("reproduction_report.txt", "w", encoding="utf-8") as f:
        f.write(f"\n{'Config':<80} | {'Paper':<10} | {'Repro':<10} | {'Diff':<10} | {'Status'}\n")
        f.write("-" * 130 + "\n")
        
        matches = 0
        mismatches = 0
        missing_claims = 0
        unverified = 0 
        
        for entry in results:
            if 'error' in entry:
                f.write(f"{str(entry.get('config', 'UNKNOWN')):<80} | ERROR\n")
                continue
                
            key = get_config_key(entry)
            repro_val = entry['mean_final_cos']
            
            # Display Name
            c = entry['config']
            ma_short = entry['model_a'].split('/')[-1]
            mb_short = entry['model_b'].split('/')[-1]
            disp_name = f"{ma_short}+{mb_short} ({c['coupling']})"
            if c['coupling'] == 'blended':
                disp_name += f" a={c['alpha']}"
            elif c['coupling'] == 'decay':
                disp_name += f" d={c['decay']}"
            if c.get('force_proj'):
                disp_name += " PROJ"
            if c['init'] != 'random':
                disp_name += f" {c['init']}"

            claim = PAPER_CLAIMS.get(key)
            
            if 'group' in c and c['group'] == 'phase_sweep':
                 f.write(f"{disp_name:<80} | {'SWEEP':<10} | {repro_val:<10.4f} | {'-':<10} | INFO\n")
                 unverified += 1
                 continue

            if claim is not None:
                target = claim
                diff = repro_val - target
                
                if abs(diff) < 0.02: 
                    status = "PASS"
                    matches += 1
                elif abs(diff) < 0.1:
                    status = "WARN"
                    mismatches += 1
                else:
                    status = "FAIL"
                    mismatches += 1
                    
                f.write(f"{disp_name:<80} | {target:<10.4f} | {repro_val:<10.4f} | {diff:<10.4f} | {status}\n")
            else:
                f.write(f"{disp_name:<80} | {'?':<10} | {repro_val:<10.4f} | {'-':<10} | UNKNOWN\n")
                missing_claims += 1

        f.write("-" * 130 + "\n")
        f.write(f"Summary: {matches} PASS, {mismatches} FAIL/WARN, {missing_claims} MISSING CLAIMS, {unverified} SWEEP RUNS\n")
    print("Report written to reproduction_report.txt")

if __name__ == "__main__":
    compare()
