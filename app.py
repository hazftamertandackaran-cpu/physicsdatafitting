import base64
import io
import json
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from regression import (
    PhysicalChebyshevRegression,
    PhysicalModelDiscovery
)


st.set_page_config(
    page_title="Physics Curve Discovery",
    page_icon="📈",
    layout="wide"
)

st.title("Physics Curve Discovery")
st.caption(
    "Chebyshev regression, cross-validation, derivatives, "
    "local scaling, and physical-law comparison."
)

st.sidebar.header("Regression settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

auto_download = st.sidebar.checkbox(
    "Auto-download results ZIP after analysis",
    value=False,
    help=(
        "Some browsers block automatic downloads. "
        "A normal Download Results ZIP button is always shown."
    )
)

# ----------------------------
# Data
# ----------------------------

if uploaded_file is None:
    st.info("Upload a CSV file or use the demo dataset.")

    use_demo = st.checkbox(
        "Use demo saturation dataset",
        value=True
    )

    if not use_demo:
        st.stop()

    np.random.seed(3)
    x_demo = np.linspace(0.5, 10, 100)
    y_demo = (
        12 * x_demo / (2.5 + x_demo)
        + 1
        + np.random.normal(0, 0.25, len(x_demo))
    )

    df = pd.DataFrame({
        "x": x_demo,
        "y": y_demo
    })
else:
    df = pd.read_csv(uploaded_file)

st.subheader("Dataset")
st.dataframe(df.head(20), use_container_width=True)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) < 2:
    st.error(
        "The CSV must contain at least two numeric columns."
    )
    st.stop()

col1, col2 = st.columns(2)

with col1:
    x_column = st.selectbox(
        "Independent variable x",
        numeric_columns,
        index=0
    )

with col2:
    possible_y = [
        c for c in numeric_columns
        if c != x_column
    ]
    y_column = st.selectbox(
        "Dependent variable y",
        possible_y,
        index=0
    )

clean = (
    df[[x_column, y_column]]
    .dropna()
    .sort_values(x_column)
)

x = clean[x_column].to_numpy(dtype=float)
y = clean[y_column].to_numpy(dtype=float)

if len(x) < 6:
    st.error("At least 6 complete observations are recommended.")
    st.stop()

max_allowed_degree = min(
    30,
    max(1, len(x) - 2)
)

max_degree = st.sidebar.slider(
    "Maximum Chebyshev degree",
    min_value=1,
    max_value=max_allowed_degree,
    value=min(12, max_allowed_degree)
)

max_folds = min(
    10,
    max(2, len(x) // 3)
)

cv_folds = st.sidebar.slider(
    "Cross-validation folds",
    2,
    max_folds,
    min(5, max_folds)
)

selection = st.sidebar.radio(
    "Degree selection",
    [
        "One-standard-error rule",
        "Maximum CV R²"
    ]
)

selection_code = (
    "one_se"
    if selection == "One-standard-error rule"
    else "max"
)

run_button = st.sidebar.button(
    "Run analysis",
    type="primary",
    use_container_width=True
)

if not run_button:
    st.stop()

# ----------------------------
# Analysis
# ----------------------------

try:
    cheb = PhysicalChebyshevRegression(
        max_degree=max_degree,
        cv_folds=cv_folds,
        selection=selection_code
    ).fit(x, y)

    discovery = PhysicalModelDiscovery(
        cv_folds=cv_folds
    ).fit(x, y)
except Exception as exc:
    st.exception(exc)
    st.stop()

st.subheader("Regression summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Selected degree", cheb.best_degree)
c2.metric(
    "CV R²",
    f"{cheb.cv_mean[cheb.best_degree]:.6f}"
)
c3.metric(
    "Training R²",
    f"{cheb.training_r2:.6f}"
)
c4.metric(
    "Residual σ",
    f"{np.sqrt(cheb.residual_variance):.6g}"
)

xx = np.linspace(np.min(x), np.max(x), 1000)
yy = cheb.predict(xx)

# ----------------------------
# Fit plot
# ----------------------------

st.subheader("Chebyshev regression")

fig, ax = plt.subplots()
ax.scatter(x, y, label="Experimental data")
ax.plot(xx, yy, label="Chebyshev reference")
ax.set_xlabel(x_column)
ax.set_ylabel(y_column)
ax.legend()
ax.grid()

st.pyplot(fig)

# ----------------------------
# CV degree optimization
# ----------------------------

st.subheader("Degree optimization")

degrees = np.arange(max_degree + 1)
se = cheb.cv_std / np.sqrt(cv_folds)

fig2, ax2 = plt.subplots()
ax2.errorbar(
    degrees,
    cheb.cv_mean,
    yerr=se,
    marker="o",
    capsize=3
)
ax2.axvline(
    cheb.best_degree,
    linestyle="--",
    label=f"Selected N={cheb.best_degree}"
)
ax2.set_xlabel("Chebyshev degree N")
ax2.set_ylabel("Cross-validated R²")
ax2.legend()
ax2.grid()

st.pyplot(fig2)

# ----------------------------
# Polynomial
# ----------------------------

st.subheader("Equivalent ordinary polynomial")

coeff = cheb.polynomial_coefficients()

poly_table = pd.DataFrame({
    "Power": np.arange(len(coeff)),
    "Coefficient": coeff
})

st.dataframe(
    poly_table,
    use_container_width=True
)

terms = []
for i, c in enumerate(coeff):
    if i == 0:
        terms.append(f"({c:.8g})")
    elif i == 1:
        terms.append(f"({c:.8g}) x")
    else:
        terms.append(f"({c:.8g}) x^{i}")

equation = " + ".join(terms)
st.code(f"y(x) = {equation}")

# ----------------------------
# Derivatives and local exponent
# ----------------------------

st.subheader("Derived physical quantities")

d1 = cheb.derivative(1)
d2 = cheb.derivative(2)

dy = d1(xx)
ddy = d2(xx)
local_n = cheb.local_exponent(xx)

tab1, tab2, tab3 = st.tabs([
    "First derivative",
    "Second derivative",
    "Local scaling exponent"
])

with tab1:
    fig3, ax3 = plt.subplots()
    ax3.plot(xx, dy)
    ax3.set_xlabel(x_column)
    ax3.set_ylabel("dy/dx")
    ax3.grid()
    st.pyplot(fig3)

with tab2:
    fig4, ax4 = plt.subplots()
    ax4.plot(xx, ddy)
    ax4.set_xlabel(x_column)
    ax4.set_ylabel("d²y/dx²")
    ax4.grid()
    st.pyplot(fig4)

with tab3:
    fig5, ax5 = plt.subplots()
    finite = np.isfinite(local_n)
    ax5.plot(xx[finite], local_n[finite])
    ax5.axhline(1, linestyle="--")
    ax5.axhline(2, linestyle="--")
    ax5.set_xlabel(x_column)
    ax5.set_ylabel("(x/y) dy/dx")
    ax5.grid()
    st.pyplot(fig5)

    st.latex(
        r"""
        n_{\mathrm{local}}(x)
        =
        \frac{x}{y}\frac{dy}{dx}
        =
        \frac{d\ln |y|}{d\ln |x|}
        """
    )

# ----------------------------
# Special points
# ----------------------------

st.subheader("Characteristic points")

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown("**Roots**")
    roots = cheb.roots()
    if len(roots):
        st.dataframe(
            pd.DataFrame({"x": roots}),
            use_container_width=True
        )
    else:
        st.write("None in measured domain.")

with p2:
    st.markdown("**Extrema**")
    extrema = cheb.extrema()
    if extrema:
        st.dataframe(
            pd.DataFrame(extrema),
            use_container_width=True
        )
    else:
        st.write("None in measured domain.")

with p3:
    st.markdown("**Inflection points**")
    inflections = cheb.inflection_points()
    if inflections:
        st.dataframe(
            pd.DataFrame(inflections),
            use_container_width=True
        )
    else:
        st.write("None in measured domain.")

# ----------------------------
# Physical model comparison
# ----------------------------

st.subheader("Physical model discovery")

rows = []

for name, result in discovery.results.items():
    if "failed" in result:
        continue

    rows.append({
        "Model": name,
        "CV R²": result["cv_r2"],
        "CV RMSE": result["cv_rmse"],
        "Training R²": result["training_r2"],
        "AICc": result["aicc"]
    })

comparison = pd.DataFrame(rows)

if not comparison.empty:
    comparison = comparison.sort_values(
        "CV R²",
        ascending=False
    )
    st.dataframe(
        comparison,
        use_container_width=True
    )
else:
    st.warning("No candidate physical model fitted successfully.")

st.caption(
    "Do not interpret the first row as a discovered physical law by itself. "
    "Use predictive performance together with parameter meaning, units, "
    "residual structure, and domain knowledge."
)

st.subheader("Candidate physical laws")

fig6, ax6 = plt.subplots()
ax6.scatter(x, y, label="Data")
ax6.plot(
    xx,
    cheb.predict(xx),
    linewidth=3,
    label="Chebyshev reference"
)

for name, result in discovery.results.items():
    if "failed" in result:
        continue
    try:
        yy_model = result["func"](
            xx,
            *result["params"]
        )
        if np.all(np.isfinite(yy_model)):
            ax6.plot(
                xx,
                yy_model,
                label=name
            )
    except Exception:
        pass

ax6.set_xlabel(x_column)
ax6.set_ylabel(y_column)
ax6.legend(fontsize=8)
ax6.grid()
st.pyplot(fig6)

# ----------------------------
# Parameter inspection
# ----------------------------

st.subheader("Physical parameters")

successful_models = [
    name
    for name, result in discovery.results.items()
    if "failed" not in result
]

parameter_table = pd.DataFrame()

if successful_models:
    selected_model = st.selectbox(
        "Inspect model",
        successful_models
    )

    result = discovery.results[selected_model]

    parameter_table = pd.DataFrame({
        "Parameter": result["param_names"],
        "Value": result["params"],
        "Standard error": result["errors"]
    })

    st.dataframe(
        parameter_table,
        use_container_width=True
    )

# ----------------------------
# Evaluate a point
# ----------------------------

st.subheader("Analyze a specific x")

x_eval = st.number_input(
    "x value",
    min_value=float(np.min(x)),
    max_value=float(np.max(x)),
    value=float(np.mean(x))
)

value = float(cheb.predict(np.array([x_eval]))[0])
slope = float(d1(x_eval))
curvature = float(d2(x_eval))

if abs(value) > 1e-15:
    scaling = float(x_eval * slope / value)
else:
    scaling = np.nan

a, b, c, d = st.columns(4)
a.metric("y(x)", f"{value:.6g}")
b.metric("dy/dx", f"{slope:.6g}")
c.metric("d²y/dx²", f"{curvature:.6g}")
d.metric("Local exponent", f"{scaling:.6g}")

# ----------------------------
# Export results
# ----------------------------

st.subheader("Export")

export_df = pd.DataFrame({
    x_column: xx,
    "Chebyshev_fit": cheb.predict(xx),
    "First_derivative": d1(xx),
    "Second_derivative": d2(xx),
    "Local_exponent": cheb.local_exponent(xx)
})

cv_df = pd.DataFrame({
    "Degree": degrees,
    "CV_R2_mean": cheb.cv_mean,
    "CV_R2_std": cheb.cv_std,
    "CV_R2_standard_error": se
})

summary = {
    "x_column": x_column,
    "y_column": y_column,
    "n_observations": int(len(x)),
    "selected_chebyshev_degree": int(cheb.best_degree),
    "selection_rule": selection,
    "chebyshev_cv_r2": float(
        cheb.cv_mean[cheb.best_degree]
    ),
    "chebyshev_training_r2": float(
        cheb.training_r2
    ),
    "residual_standard_deviation": float(
        np.sqrt(cheb.residual_variance)
    ),
    "ordinary_polynomial_coefficients": [
        float(v) for v in coeff
    ],
    "roots": [float(v) for v in roots],
    "extrema": extrema,
    "inflection_points": inflections
}

model_parameters_rows = []
for name, result in discovery.results.items():
    if "failed" in result:
        continue
    for pname, pval, perr in zip(
        result["param_names"],
        result["params"],
        result["errors"]
    ):
        model_parameters_rows.append({
            "Model": name,
            "Parameter": pname,
            "Value": float(pval),
            "Standard_error": float(perr)
        })

model_parameters_df = pd.DataFrame(
    model_parameters_rows
)

# Build a result ZIP in memory.
result_zip_buffer = io.BytesIO()

with zipfile.ZipFile(
    result_zip_buffer,
    mode="w",
    compression=zipfile.ZIP_DEFLATED
) as zf:
    zf.writestr(
        "processed_curve.csv",
        export_df.to_csv(index=False)
    )
    zf.writestr(
        "degree_cross_validation.csv",
        cv_df.to_csv(index=False)
    )
    zf.writestr(
        "polynomial_coefficients.csv",
        poly_table.to_csv(index=False)
    )
    zf.writestr(
        "physical_model_comparison.csv",
        comparison.to_csv(index=False)
        if not comparison.empty else ""
    )
    zf.writestr(
        "physical_model_parameters.csv",
        model_parameters_df.to_csv(index=False)
    )
    zf.writestr(
        "analysis_summary.json",
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )
    zf.writestr(
        "source_data.csv",
        clean.to_csv(index=False)
    )

result_zip_bytes = result_zip_buffer.getvalue()

st.download_button(
    "Download Results ZIP",
    data=result_zip_bytes,
    file_name="physics_fit_results.zip",
    mime="application/zip",
    type="primary"
)

st.download_button(
    "Download processed curve CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="physics_fit_results.csv",
    mime="text/csv"
)

# Optional browser-side automatic download.
# This can be blocked by browser security settings, so the normal button remains.
if auto_download:
    payload = base64.b64encode(
        result_zip_bytes
    ).decode("ascii")

    components.html(
        f"""
        <html>
        <body>
        <script>
        const a = document.createElement('a');
        a.href = 'data:application/zip;base64,{payload}';
        a.download = 'physics_fit_results.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        </script>
        <p>Automatic download was requested. If your browser blocked it,
        use the Download Results ZIP button above.</p>
        </body>
        </html>
        """,
        height=70
    )
