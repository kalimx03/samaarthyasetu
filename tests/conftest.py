"""
conftest.py — adds project root to sys.path for all test files.
This replaces the sys.path.insert() boilerplate in each test module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
