"""Train a two-program policy with live CUDA measurements and source actions.

The policy and Gym training loop live in gpu2tensor.examples.learn.
Run this file with --help for worker, learner device and episode options.
"""

from gpu2tensor.examples.learn import main


if __name__ == "__main__":
    main()
