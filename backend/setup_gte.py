#!/usr/bin/env python3
"""
Setup script to install required packages for GTE-multilingual embedding provider
"""
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        "sentence-transformers",
        "torch"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

if __name__ == "__main__":
    success = install_packages()
    if success:
        print("🎉 All packages installed successfully!")
    else:
        print("❌ Some packages failed to install")
    sys.exit(0 if success else 1)