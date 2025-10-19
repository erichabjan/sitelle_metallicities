import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description="Script that accepts a string input.")
parser.add_argument("input_string", type=str, help="The input string to process")
args = parser.parse_args()
galaxy = args.input_string

#scripts = ["montage_reproject.py", "emission_line_fitting.py", "physical_quantities.py", "strong_calibration.py"]
scripts = ["physical_quantities.py", "strong_calibration.py"]

job_log_path = os.path.join('/home/habjan/SITELLE/sandbox_notebooks/job_logs', galaxy)
os.makedirs(job_log_path, exist_ok=True)

dirc_path = '/home/habjan/SITELLE/sitelle_metallicities/'

for script in scripts:
    print(f"Running {script}")
    
    # Build the log file path
    log_file = os.path.join(job_log_path, f"{script}.log")
    
    script_path = dirc_path + script
    
    # Build the command
    if script == "strong_calibration.py":
        cmd = f"python {script_path} > {log_file} 2>&1"
    else:
        cmd = f"python {script_path} {galaxy} > {log_file} 2>&1"
    
    # Run and wait for it to finish
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    
    print(f"Finished {script}\n")

print("All scripts executed.")

