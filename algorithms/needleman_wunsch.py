"""
A dynamic programming algorithm for optimal global alignment of two sequences:
1. Build a (len(seq1)+1)×(len(seq2)+1) score matrix, initializing with gap penalties.
2. Fill the matrix by choosing the maximum of:
   • diagonal (match_score or mismatch_penalty)
   • up (gap_penalty for a gap in seq2)
   • left (gap_penalty for a gap in seq1)
3. Trace back from the bottom-right to reconstruct the aligned sequences.
"""
import numpy as np
from matplotlib import pyplot as plt


class NeedlemanWunsch:
    """
    Perform global alignment of two sequences using Needleman–Wunsch.

    Parameters:
        seq1 – first input sequence
        seq2 – second input sequence
        match_score – score added when characters match
        mismatch_penalty – penalty added when characters mismatch
        gap_penalty – penalty added when inserting a gap

        score_matrix – 2D list of alignment scores
        aligned_seq1 – resulting aligned version of seq1
        aligned_seq2 – resulting aligned version of seq2
        final_score – optimal alignment score (bottom-right of matrix)
        traceback_path     – list of (i,j) coords visited during traceback
    """
    def __init__(self, seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
        self.seq1 = seq1
        self.seq2 = seq2
        self.match = match_score
        self.mismatch = mismatch_penalty
        self.gap = gap_penalty

        self.score_matrix = []
        self.aligned_seq1 = ""
        self.aligned_seq2 = ""
        self.final_score = None
        self.traceback_path = []

    def _initialize_matrix(self):
        n, m = len(self.seq1), len(self.seq2)
        # create zero matrix
        self.score_matrix = [[0] * (m + 1) for _ in range(n + 1)]
        # first column: gaps in seq2
        for i in range(1, n + 1):
            self.score_matrix[i][0] = self.score_matrix[i - 1][0] + self.gap
        # first row: gaps in seq1
        for j in range(1, m + 1):
            self.score_matrix[0][j] = self.score_matrix[0][j - 1] + self.gap

    def _fill_matrix(self):
        n, m = len(self.seq1), len(self.seq2)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # diagonal: match or mismatch
                if self.seq1[i - 1] == self.seq2[j - 1]:
                    diag = self.score_matrix[i - 1][j - 1] + self.match
                else:
                    diag = self.score_matrix[i - 1][j - 1] + self.mismatch
                # gap in seq2 (move down)
                down = self.score_matrix[i - 1][j] + self.gap
                # gap in seq1 (move right)
                right = self.score_matrix[i][j - 1] + self.gap
                # choose best
                self.score_matrix[i][j] = max(diag, down, right)

    def _traceback(self):
        i, j = len(self.seq1), len(self.seq2)
        aligned1, aligned2 = [], []
        path = []
        while i>0 or j>0:
            path.append((i, j))
            current = self.score_matrix[i][j]
            # diagonal
            if i>0 and j>0:
                score_diag = self.score_matrix[i-1][j-1] + (
                    self.match if self.seq1[i-1]==self.seq2[j-1] else self.mismatch
                )
                if current == score_diag:
                    aligned1.append(self.seq1[i-1])
                    aligned2.append(self.seq2[j-1])
                    i, j = i-1, j-1
                    continue
            # up (gap in seq2)
            if i>0 and current == self.score_matrix[i-1][j] + self.gap:
                aligned1.append(self.seq1[i-1])
                aligned2.append('-')
                i -= 1
                continue
            # left (gap in seq1)
            aligned1.append('-')
            aligned2.append(self.seq2[j-1])
            j -= 1
        path.append((0, 0))
        self.traceback_path = path[::-1]  # reverse for start→end
        self.aligned_seq1 = ''.join(reversed(aligned1))
        self.aligned_seq2 = ''.join(reversed(aligned2))

    def run(self):
        """
        Execute the alignment:
        1. Initialize score matrix
        2. Fill matrix with dynamic programming
        3. Trace back to build aligned sequences
        """
        self._initialize_matrix()
        self._fill_matrix()
        self.final_score = self.score_matrix[len(self.seq1)][len(self.seq2)]
        self._traceback()

    def plot_traceback(self):
        """
        Plot the score matrix as a heatmap and overlay the traceback path.
        """
        if not self.score_matrix or not self.traceback_path:
            raise RuntimeError("Call run() before plot_traceback()")

        mat = np.array(self.score_matrix)
        n, m = len(self.seq1), len(self.seq2)

        plt.figure(figsize=(8, 6))
        plt.imshow(mat, cmap='viridis', origin='upper')
        plt.colorbar(label='Score')

        # ticks for sequences (include gap at index 0)
        xt = np.arange(m + 1)
        yt = np.arange(n + 1)
        plt.xticks(xt, ['-'] + list(self.seq2), rotation=90)
        plt.yticks(yt, ['-'] + list(self.seq1))

        # path coordinates: rows = i, cols = j
        path = np.array(self.traceback_path)
        plt.plot(path[:, 1], path[:, 0], '-o', color='red')

        plt.gca().invert_yaxis()
        plt.title('Needleman–Wunsch Score Matrix & Traceback Path')
        plt.xlabel('Sequence 2')
        plt.ylabel('Sequence 1')
        plt.tight_layout()
        plt.show()