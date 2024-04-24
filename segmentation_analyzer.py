# utils/segmentation_analyzer.py

import pandas as pd

class SegmentationAnalyzer:
    def __init__(self, data):
        self.data = data

    def count_segments(self, column):
        """Count segments based on a specific column"""
        return self.data[column].value_counts()
