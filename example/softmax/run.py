"""Check a complete softmax kernel on CPU, CUDA or Trainium.

The implementation and submitted kernels live in gpu2tensor.examples.softmax.
Run this file with --help for local and remote options.
"""

from gpu2tensor.examples.softmax import main


if __name__ == "__main__":
    main()
