"""Install NumPy using pip: pip install numpy"""

import subprocess
import sys

def check_numpy_installed():
    try:
        import numpy
        return True, numpy.__version__
    except ImportError:
        return False, None

def install_numpy():
    print("Installing NumPy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

if __name__ == "__main__":
    installed, version = check_numpy_installed()
    if installed:
        print(f"NumPy already installed. Version: {version}")
    else:
        print("NumPy not found. Installing...")
        install_numpy()
        installed, version = check_numpy_installed()
        print(f"NumPy installed successfully. Version: {version}")