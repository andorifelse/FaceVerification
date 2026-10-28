"""Default training entry point: fit and select the refined Transformer on CASIA.

The previous convolutional baseline training remains in train_mobile.py.
"""
import sys
from scripts.run_refinement import main

if __name__ == '__main__':
    sys.argv.append('--train-only')
    main()
