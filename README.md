# Gramba

## Requirements 
- Conda

## Setup
1) Clone the repository onto your machine (note that the conda environment was created on the U of T servers).
2) Create the virtual environment: ```conda env create -f environment.yml```
> **Note:** This may take several minutes.
3) Activate the environment: ```conda activate gramba```
4) Ensure the src directory is in your Python Path: ```export PYTHONPATH="$PYTHONPATH:/path/to/repository/src"```
> **Note:** This will only add the src directory to your path for your current session.  Write the export command to your .bashrc, .zshrc, or equivalent to add the path permanently.
5) Verify that test cases work: ```pytest```
