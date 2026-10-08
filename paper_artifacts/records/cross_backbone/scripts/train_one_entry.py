"""Entry point for the immutable cross-backbone job runner."""
import train_one
from cross_backbone_common import NUM_CLASSES

train_one.NUM_CLASSES = NUM_CLASSES

if __name__ == "__main__":
    train_one.main()
