#!/usr/bin/env python3
"""Compare results across validation runs."""
import json
import sys
import glob
import os

def load_latest(directory):
    """Load the latest full_validation JSON from a directory."""
    pattern = os.path.join(directory, "full_validation_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

def main():
    dirs = [
        ("Run 1 (42,43,44)", "data/results"),
        ("Run 2 (100,101,102)", "data/results_run2"),
        ("Run 3 (200,201,202)", "data/results_run3"),
    ]
    
    runs = {}
    for name, d in dirs:
        r = load_latest(d)
        if r:
            runs[name] = r
            print(f"Loaded {name} from {d}")
            
            # Load additional dynamics (separate files)
            dyn_pattern = os.path.join(d, "dynamics_*.json")
            dyn_files = sorted(glob.glob(dyn_pattern))
            for df in dyn_files:
                try:
                    with open(df) as f:
                        dyn_data = json.load(f)
                    if dyn_data.get('experiment') == 'coupled_dynamics':
                        # Check for finetuning_medium (gpt2-medium + DialoGPT-medium)
                        if dyn_data.get('model_a') == 'gpt2-medium' and 'DialoGPT-medium' in dyn_data.get('model_b', ''):
                            # Inject into the main results dict
                            if 'dynamics' not in runs[name]:
                                runs[name]['dynamics'] = {}
                            runs[name]['dynamics']['finetuning_medium'] = dyn_data
                            print(f"  + Found finetuning_medium dynamics in {os.path.basename(df)}")
                except Exception as e:
                    print(f"  Error reading {df}: {e}")

        else:
            print(f"No results yet for {name}")
    
    if len(runs) < 2:
        print("\nNeed at least 2 runs to compare. Exiting.")
        return
    
    # Dynamics comparison
    print("\n" + "=" * 90)
    print("DYNAMICS (cos theta)")
    print("=" * 90)
    header = f"{'Config':<20}"
    for name in runs:
        header += f" | {name:<25}"
    print(header)
    print("-" * 90)
    
    for key in ['identity', 'distillation', 'finetuning_small', 'finetuning_medium', 'cross_scale']:
        row = f"{key:<20}"
        for name, r in runs.items():
            d = r.get('dynamics', {}).get(key, {})
            cos = d.get('mean_final_cos')
            std = d.get('std_final_cos')
            if cos is not None:
                row += f" | {cos:.4f} +/- {std:.4f}       "
            else:
                row += f" | {'N/A':<25}"
        print(row)
    
    # CKA comparison
    print("\n" + "=" * 90)
    print("CKA TABLE")
    print("=" * 90)
    
    first_run = list(runs.values())[0]
    pairs = first_run.get('cka', {}).get('pairs', [])
    
    for i, p in enumerate(pairs):
        pair_name = p['pair']
        print(f"\n  {pair_name}:")
        for name, r in runs.items():
            rp = r.get('cka', {}).get('pairs', [])
            if i < len(rp):
                print(f"    {name}: CKA={rp[i]['cka']:.4f}  cos_theta={rp[i]['cos_theta']:.4f} +/- {rp[i]['cos_theta_std']:.4f}")
    
    # Merge comparison
    print("\n" + "=" * 90)
    print("WEIGHT MERGING (GPT-2-Med + DialoGPT-Med)")
    print("=" * 90)
    
    for name, r in runs.items():
        m = r.get('merge', {})
        base = m.get('ppl_a', 0)
        print(f"\n  {name}: baseline PPL = {base:.2f}")
        merge_results = m.get('merge_results', {})
        if isinstance(merge_results, dict):
            for alpha_key, mr in sorted(merge_results.items()):
                ppl = mr.get('ppl_merged', 0)
                pct = mr.get('degradation_pct', 0)
                print(f"    alpha={alpha_key}: PPL={ppl:.2f} (+{pct:.1f}%%)")
        else:
            print("    (merge results in unexpected format)")
    
    # Consistency check
    print("\n" + "=" * 90)
    print("CONSISTENCY VERDICT")
    print("=" * 90)
    
    run_list = list(runs.values())
    run_names = list(runs.keys())
    
    all_consistent = True
    
    for key in ['identity', 'distillation', 'finetuning_small', 'cross_scale']:
        cos_values = []
        for r in run_list:
            d = r.get('dynamics', {}).get(key, {})
            cos = d.get('mean_final_cos')
            if cos is not None:
                cos_values.append(cos)
        
        if len(cos_values) >= 2:
            spread = max(cos_values) - min(cos_values)
            mean = sum(cos_values) / len(cos_values)
            status = "CONSISTENT" if spread < 0.05 else "VARIABLE"
            if status == "VARIABLE":
                all_consistent = False
            print(f"  {key:<20} spread={spread:.4f}  mean={mean:.4f}  [{status}]")
    
    if all_consistent:
        print("\n  >>> ALL RESULTS CONSISTENT ACROSS RUNS <<<")
    else:
        print("\n  >>> SOME VARIABILITY DETECTED - review above <<<")

if __name__ == '__main__':
    main()
