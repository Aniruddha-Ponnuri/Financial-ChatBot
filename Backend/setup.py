#!/usr/bin/env python3
"""
Setup script for Financial Chatbot Backend
Installs all required dependencies
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print("=" * 60)
    try:
        subprocess.check_call(command, shell=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}")
        return False


def main():
    print("=" * 60)
    print("  Financial Chatbot - Backend Setup")
    print("=" * 60)

    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher required")
        sys.exit(1)

    # Upgrade pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("Warning: Could not upgrade pip, continuing anyway...")

    # Install from requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

    if os.path.exists(requirements_path):
        print(f"\nInstalling from requirements.txt...")
        if not run_command(
            f"{sys.executable} -m pip install -r {requirements_path}",
            "Installing dependencies",
        ):
            print("\nSome packages failed to install.")
            print("You can install them manually using:")
            print(f"  pip install -r {requirements_path}")
    else:
        print(f"\nWARNING: requirements.txt not found at {requirements_path}")
        print("Installing core dependencies manually...")

        # Core dependencies
        packages = [
            "flask flask-cors python-dotenv pyyaml",
            "langchain langchain-core",
            "langchain-openai langchain-groq",
            "torch transformers sentence-transformers",
            "yfinance numpy",
        ]

        for package_group in packages:
            run_command(
                f"{sys.executable} -m pip install {package_group}",
                f"Installing {package_group.split()[0]}",
            )

    print("\n" + "=" * 60)
    print("  Setup completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Configure your .env file with API keys")
    print("2. Run: python app.py")
    print("\nSee LLM_PROVIDER_GUIDE.md for configuration details")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
