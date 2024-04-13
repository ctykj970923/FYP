import os
import subprocess
import sys
def create_requirements_file(output_file='requirements.txt'):
    """
    Creates a requirements.txt file containing the installed packages and their versions.
    
    Parameters:
    output_file (str): The name of the output file (default is 'requirements.txt').
    """
    
    # Check if pip is installed
    try:
        import pip
    except ImportError:
        print("Error: pip is not installed.")
        return
    
    # Get the list of installed packages and their versions
    installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode('utf-8').strip().split('\n')
    
    # Write the requirements to the output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(installed_packages))
    
    print(f"Requirements file '{output_file}' created successfully.")

# Example usage
create_requirements_file()