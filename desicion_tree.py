import helper_functions as hf
import numpy as np
import pandas as pd


if __name__ == "__main__":
    print(hf.HasMissingValue("income"))
    values = hf.ExtractValues("income")
    print(len(values))
