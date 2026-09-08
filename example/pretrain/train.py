"""Train a tiny classifier from recorded or freshly collected GPU profiles.

The collection, model and training loop live in gpu2tensor.examples.pretrain.
Run this file with --help for the dataset, live endpoint and offline reuse modes.
"""

from gpu2tensor.examples.pretrain import main


if __name__ == "__main__":
    main()
