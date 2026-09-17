# Physics Curve Discovery

A Streamlit interface for:

- Chebyshev least-squares regression
- degree selection using cross-validated R²
- one-standard-error model selection
- conversion to an ordinary polynomial
- first and second derivatives
- roots, extrema, and inflection points
- local logarithmic scaling exponent
- comparison with simple candidate physical laws
- AICc, CV R², and CV RMSE
- parameter uncertainties
- one-click ZIP export of analysis results
- optional browser-side automatic results download

## Quick start

### Windows

Double-click:

    start_windows.bat

or run:

    pip install -r requirements.txt
    streamlit run app.py

### macOS / Linux

Run:

    chmod +x start_mac_linux.sh
    ./start_mac_linux.sh

or manually:

    pip install -r requirements.txt
    streamlit run app.py

## Input CSV

The CSV needs at least two numeric columns.

Example:

    x,y
    0.5,2.1
    1.0,3.4
    1.5,4.2

You choose which numeric column is x and which is y in the interface.

## About automatic downloading

The sidebar contains:

    Auto-download results ZIP after analysis

When enabled, the app tries to trigger a browser download after analysis. Some browsers or security settings may block this behavior. The normal "Download Results ZIP" button is always available and is more reliable.

## Important scientific note

The physical-law comparison is a model-screening tool, not proof of a governing law. Prefer models that are:

1. predictive out of sample,
2. physically interpretable,
3. dimensionally meaningful,
4. supported by residual analysis,
5. appropriate to the experimental regime.

A slightly lower CV R² can still be preferable scientifically if the model has a clear mechanistic interpretation.
