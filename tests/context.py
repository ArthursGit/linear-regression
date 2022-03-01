import sys
import os
print("context.py")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import linear_regression
import datasets