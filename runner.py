import subprocess
import argparse

parser = argparse.ArgumentParser(description="Script that accepts a string input.")
parser.add_argument("input_string", type=str, help="The input string to process")
args = parser.parse_args()
galaxy = args.input_string

scripts = ["montage_reproject.py", "emission_line_fitting.py", "physical_quantities.py", "strong_calibration.py"]
job_log_path = '/home/habjan/SITELLE/sandbox_notebooks/job_logs'

for script in scripts:
    
    if script == "strong_calibration.py": 
        
        print(f"Running {script}")
        log_file = f"{job_log_path + script}.log"
        process = subprocess.Popen(f"nohup python {script} > {log_file} 2>&1 &", shell=True)
        process.wait()
        print(f"Finished {script}\n")
    
    else: 
        
        print(f"Running {script}")
        log_file = f"{job_log_path + script}.log"
        process = subprocess.Popen(f"nohup python {script} {galaxy} > {log_file} 2>&1 &", shell=True)
        process.wait()
        print(f"Finished {script}\n")

print("All scripts executed.")

