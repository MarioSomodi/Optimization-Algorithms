import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.needleman_wunsch import NeedlemanWunsch

class NeedlemanWunschTimeComplexityEvaluator:
    """
    Evaluate and demonstrate the time complexity of the Needleman–Wunsch algorithm.
    """
    def __init__(self, lengths, runs=5, seed=None):
        """
            lengths: sequence lengths to test (will use seq1, seq2 of length L)
            runs: number of repetitions per length to average out noise
            seed: random seed for reproducibility of sequences
            raw_data: dict with lists 'L', 'm·n', 'avg_time_s'
            df: pandas DataFrame with columns ['L', 'm·n', 'avg_time_s']
        """
        self.lengths = lengths
        self.runs = runs
        self.seed = seed

        self.raw_data = None
        self.df = None

    def evaluate(self):
        """
        For each L in self.lengths, generate two random DNA sequences of length L,
        run NeedlemanWunsch.run(), time it, and record L, L·L, and avg time.
        """
        rng = random.Random(self.seed)
        records = []

        for L in self.lengths:
            times = []
            for _ in range(self.runs):
                # generate two random DNA strings
                seq1 = ''.join(rng.choice('ACGT') for _ in range(L))
                seq2 = ''.join(rng.choice('ACGT') for _ in range(L))
                # time the full alignment (init + fill + traceback)
                t0 = time.perf_counter()
                nw = NeedlemanWunsch(seq1, seq2,
                                     match_score=1,
                                     mismatch_penalty=-1,
                                     gap_penalty=-2)
                nw.run()
                times.append(time.perf_counter() - t0)

            records.append({
                'L': L,
                'm·n': L * L,
                'avg_time_s': sum(times) / len(times)
            })

        self.raw_data = {k: [r[k] for r in records] for k in records[0]}
        self.df = pd.DataFrame(records)

    def plot_complexity(self):
        """
        Plot avg_time_s vs. m·n and compare to the ideal quadratic curve.
        """
        if self.df is None:
            raise RuntimeError("Call evaluate() before plotting.")

        x = self.df['m·n']
        y = self.df['avg_time_s']

        # normalize so that the quadratic reference has similar scale
        coeff = y.iloc[-1] / x.iloc[-1]

        plt.figure(figsize=(7,5))
        plt.plot(x, y, 'o-', label='Measured runtime')
        plt.plot(x, coeff * x, '--', label=r'c·$L^2$ reference')
        plt.xlabel('L·L (matrix cells)')
        plt.ylabel('Average time (s)')
        plt.title('Needleman–Wunsch Runtime vs. Problem Size')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
