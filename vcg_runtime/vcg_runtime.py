# @title Setup Constants

import numpy as np

np.random.seed(42)

# @title Generate Bids and Preference
def generate_bids(n_bidders: int, n_genres: int):
    """
    Generates random float bids with a Poisson distribution
    for the number of bidding genres per bidder, where bids are uniformly distributed in [0, 1].

    Args:
        n_bidders: The number of bidders.
        n_genres: The number of ad genres.

    Returns:
        preferences: A numpy array of random float bids.
    """

    bids = np.zeros((n_bidders, n_genres)) # Default float

    for i in range(n_bidders):
        # Generate number of preferred genres using Poisson distribution, ensuring at least 1
        num_preferred_genres = np.random.poisson(lam=2)
        num_preferred_genres = min(max(1, num_preferred_genres),n_genres)

        # Randomly select preferred genres
        preferred_genres_indices = np.random.choice(n_genres, size=num_preferred_genres, replace=False)
        # Assign random values in [0, 1]
        bids[i, preferred_genres_indices] = np.random.rand(num_preferred_genres)

    return bids


# @title Generate Coherence Matrix
def generate_coherence_matrix(n_slots, n_ad_genre, dummy_coherence=True, user_prompt=None, ad_genre_list=None, organic_response=None):
    """
    Generates a coherence matrix between ad slots and genres.

    Args:
        n_slots: The number of ad slots.
        n_ad_genre: The number of ad genres.
        dummy_coherence: If True, generates a dummy coherence matrix with random values.
                         If False, the coherence matrix is initialized with zeros and can be
                         updated based on user_prompt and organic_response.
        ad_genre_list: A list of ad genres.
        user_prompt: The user's search query (not used in dummy generation).
        organic_response: The organic search result (not used in dummy generation).

    Returns:
        A numpy array representing the coherence matrix.
    """
    coherence_matrix = np.zeros((n_slots, n_ad_genre))
    if dummy_coherence:
        # Generate dummy coherence values as floats between 0 and 1
        coherence_matrix = np.random.rand(n_slots, n_ad_genre)

        # Set some values to a very low number to represent low coherence
        low_coherence_indices = np.random.choice(n_slots * n_ad_genre, size=int(0.5 * n_slots * n_ad_genre), replace=False)
        coherence_matrix.flat[low_coherence_indices] = 0

        # Ensure each row has at least one value not equal to 0
        for i in range(n_slots):
            if np.all(coherence_matrix[i, :] == 0):
                random_genre_index = np.random.randint(n_ad_genre)
                coherence_matrix[i, random_genre_index] = np.random.rand()

    else:
        pass

    return coherence_matrix


# @title Compute Matching Matrix
def compute_matching_matrix(bids, coherence_matrix):
    """
    Computes the matching matrix based on bids and coherence matrix.
    Formula: matching_matrix[i, j] = sum_k(coherence[j, k] * bids[i, k])

    Args:
        bids: A numpy array of bids over each genre, shape = (n_bidders, n_genres).
        coherence_matrix: A numpy array representing the coherence matrix, shape = (n_slots, n_genres).

    Returns:
        A numpy array representing the matching matrix, shape = (n_bidders, n_slots).
    """
    matching_matrix = np.dot(bids, coherence_matrix.T)

    return matching_matrix


# @title VCG Mechanism with JV

from scipy.optimize import linear_sum_assignment

def vcg_assignment(V, K):
    """
    VCG (Clarke pivot) payments for one-to-one assignment.

    Args:
      V: (n_bidders x n_items) valuations; no -inf entries (all pairs allowed).
         Unmatched = outside option 0.
      K: exactly assign K items (must satisfy 0 <= K <= min(n_bidders, n_items)).

    Returns:
      matches: list of length n_bidders with item index or None (if unmatched)
      payments: list of length n_bidders with VCG payment for each bidder
      welfare: total welfare of allocation (sum of assigned bidders' valuations)
    """
    V = np.asarray(V, dtype=float)
    n_bidders, n_items = V.shape

    def solve_exact_k(V, K):
        """
        Solve welfare-maximizing assignment selecting exactly K real items,
        """
        n_bidders, n_items = V.shape

        if min(n_bidders,n_items) > K:
            V_pad = np.concatenate([V, np.ones((min(n_bidders,n_items) - K, n_items)) * 1000000000], axis = 0)
        else:
            V_pad = V

        row_ind, col_ind = linear_sum_assignment(-V_pad)

        # Extract matches for REAL bidders to REAL items only
        matches = [None] * n_bidders
        welfare = 0.0
        for r, c in zip(row_ind, col_ind):
            if r < n_bidders and c < n_items:
                matches[r] = c
                welfare += V[r, c]
        return matches, welfare

    # --- 1) Welfare-maximizing allocation with all bidders (exactly K items)
    matches, welfare = solve_exact_k(V, K)

    # --- 2) Compute VCG payments (Clarke pivot)
    payments = [0.0] * n_bidders
    # Value each bidder gets in the chosen allocation (0 if unmatched)
    bidder_value = [0.0] * n_bidders
    for i, j in enumerate(matches):
        if j is not None:
            bidder_value[i] = V[i, j]

    # Total welfare with everyone:
    W = welfare

    for i in range(n_bidders):
        if matches[i] is None:
            payments[i] = 0.0
            continue

        # Remove bidder i and recompute optimal welfare for others
        V_minus_i = V.copy()
        V_minus_i[i,:] -= 1000000000
        _, W_minus_i = solve_exact_k(V_minus_i, K)

        # Clarke pivot: payment = (welfare of others without i) - (welfare of others with i)
        payments[i] = W_minus_i - (W - bidder_value[i])

    return matches, payments, welfare


# @title Running

import time
import os
import numpy as np
import pandas as pd


if "__file__" in globals():
    working_dir = os.path.dirname(os.path.abspath(__file__))
else:
    working_dir = os.getcwd()
working_dir = os.path.join(working_dir, "")


def run_experiment(n_bidders: int, n_genres: int, n_slots: int, K: int, num_runs: int):
    """
    Runs num_runs trials for a given (n_bidders, n_slots) and returns runtimes + welfare.
    """

    runtimes = []
    welfares = []

    for t in range(num_runs):
        # Generate new data for each run
        bids = generate_bids(n_bidders, n_genres)
        coherence_matrix = generate_coherence_matrix(n_slots, n_genres)

        # Measure VCG runtime and welfare
        start_time = time.perf_counter()
        V = compute_matching_matrix(bids, coherence_matrix)
        _, _, welfare = vcg_assignment(V=V, K=K)
        end_time = time.perf_counter()

        runtimes.append(end_time - start_time)
        welfares.append(welfare)

    return np.array(runtimes), np.array(welfares)

n_bidders_list = [1000, 10000, 100000]
n_slots_list   = [20, 50, 100]

n_genres = 100
K = 5
num_runs = 100

all_rows = []

for n_bidders in n_bidders_list:
    for n_slots in n_slots_list:
        print(f"Running config: n_bidders={n_bidders}, n_slots={n_slots}, n_genres={n_genres}, K={K}, num_runs={num_runs}")

        runtimes, welfares = run_experiment(
            n_bidders=n_bidders,
            n_genres=n_genres,
            n_slots=n_slots,
            K=K,
            num_runs=num_runs,
        )

        all_rows.append({
            "n_bidders": n_bidders,
            "n_slots": n_slots,
            "n_genres": n_genres,
            "K": K,
            "num_runs": num_runs,
            "runtime_mean_s": float(np.mean(runtimes)),
            'runtimes': runtimes,
            'welfares': welfares,
        })

# ---- Summary table ----
summary_df = pd.DataFrame(all_rows).sort_values(["n_bidders", "n_slots"]).reset_index(drop=True)

# Display nicely
print("\n=== Runtime Summary (seconds) ===")
print(summary_df) 

# save summary
summary_df.to_csv(working_dir + "vcg_runtime_summary.csv", index=False)
