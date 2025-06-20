def needleman_wunsch(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
    """
    Perform global sequence alignment using the Needleman-Wunsch algorithm.

    Args:
        seq1 (str): First sequence.
        seq2 (str): Second sequence.
        match_score (int): Score for a character match.
        mismatch_penalty (int): Penalty for a character mismatch.
        gap_penalty (int): Penalty for introducing a gap.

    Returns:
        aligned_seq1 (str): Aligned version of seq1.
        aligned_seq2 (str): Aligned version of seq2.
        final_score (int): Optimal alignment score.
    """
    # lengths
    n, m = len(seq1), len(seq2)

    # 1. Initialization: create score matrix of size (n+1)x(m+1)
    score = [[0] * (m + 1) for _ in range(n + 1)]

    # initialize first column (all gaps in seq2)
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + gap_penalty

    # initialize first row (all gaps in seq1)
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + gap_penalty

    # 2. Filling the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # match or mismatch
            if seq1[i - 1] == seq2[j - 1]:
                diag = score[i - 1][j - 1] + match_score
            else:
                diag = score[i - 1][j - 1] + mismatch_penalty

            # gap in seq1 (move down)
            down = score[i - 1][j] + gap_penalty
            # gap in seq2 (move right)
            right = score[i][j - 1] + gap_penalty

            # choose the maximum
            score[i][j] = max(diag, down, right)

    # 3. Traceback to build aligned sequences
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n, m

    while i > 0 or j > 0:
        current = score[i][j]

        # check coming from diagonal (match/mismatch)
        if i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                score_diag = score[i - 1][j - 1] + match_score
            else:
                score_diag = score[i - 1][j - 1] + mismatch_penalty
            if current == score_diag:
                aligned_seq1.append(seq1[i - 1])
                aligned_seq2.append(seq2[j - 1])
                i -= 1
                j -= 1
                continue

        # check coming from above (gap in seq2)
        if i > 0 and current == score[i - 1][j] + gap_penalty:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append('-')
            i -= 1
            continue

        # otherwise, must be from left (gap in seq1)
        if j > 0 and current == score[i][j - 1] + gap_penalty:
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j - 1])
            j -= 1
            continue

        # safety break (should not happen)
        break

    # reverse to get correct order
    aligned_seq1 = ''.join(reversed(aligned_seq1))
    aligned_seq2 = ''.join(reversed(aligned_seq2))
    final_score = score[n][m]

    return aligned_seq1, aligned_seq2, final_score