"""Numbers quoted from Karaman (2023), Appl. Sci. 13, 11455, Table 5 (test set).

The paper's networks are NOT re-implemented here; these published values are the
comparison target. They were obtained on a random 75/25 split of the cleaned
hourly rows, with errors on the min-max scale (0-1) of active power.
"""

from __future__ import annotations

import pandas as pd

PAPER_CITATION = "Karaman (2023), Appl. Sci. 13, 11455, Table 5"

PAPER_TABLE5 = pd.DataFrame(
    {
        "model": ["ANN", "CNN", "RNN", "LSTM"],
        "mae_scaled": [0.0245, 0.0235, 0.0218, 0.0209],
        "mse_scaled": [0.0062, 0.0055, 0.0043, 0.0038],
        "rmse_scaled": [0.0787, 0.0742, 0.0656, 0.0614],
        "r2": [0.9301, 0.9378, 0.9514, 0.9574],
    }
)

PAPER_BEST = "LSTM"
