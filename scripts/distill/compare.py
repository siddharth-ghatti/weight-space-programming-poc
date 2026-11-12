"""Write runs/d1/compare.md from every arm's eval/scores.json plus the existing rows of runs/compare.md."""
from wspd.eval.compare import write_compare

if __name__ == "__main__":
    print(write_compare())
