import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.summarizer import create_summary

summary = create_summary()

print("\nSUMMARY:\n")
print(summary)
