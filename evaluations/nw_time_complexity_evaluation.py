import time
import random
import pandas as pd
from algorithms.needleman_wunsch import needleman_wunsch

def evaluate_nw_time_complexity(lengths, runs=5):
    """
    Measure average runtime of needleman_wunsch(seq1, seq2) over random DNA sequences of length L.
    Returns a DataFrame with columns: L, m·n, avg_time_s.
    """
    records = []
    for L in lengths:
        # generate two random DNA strings of length L
        seq1 = ''.join(random.choice('ACGT') for _ in range(L))
        seq2 = ''.join(random.choice('ACGT') for _ in range(L))

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            # full run: fill + traceback
            needleman_wunsch(seq1, seq2,
                             match_score=1,
                             mismatch_penalty=-1,
                             gap_penalty=-2)
            times.append(time.perf_counter() - t0)

        records.append({
            'L': L,
            'm·n': L * L,
            'avg_time_s': sum(times) / len(times)
        })

    return pd.DataFrame(records)