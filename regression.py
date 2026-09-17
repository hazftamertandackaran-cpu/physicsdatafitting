import numpy as np
import warnings
from numpy.polynomial import Chebyshev, Polynomial
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class PhysicalChebyshevRegression:
    def __init__(self, max_degree=20, cv_folds=5, selection="one_se"):
        self.max_degree = int(max_degree)
        self.cv_folds = int(cv_folds)
        self.selection = selection

        self.best_degree = None
        self.cheb_model = None
        self.poly_model = None
        self.cv_mean = None
        self.cv_std = None
        self.xmin = None
        self.xmax = None
        self.training_r2 = None
        self.residual_variance = None

    def _fit_degree(self, x, y, degree):
        return Chebyshev.fit(
            x, y, degree,
            domain=[self.xmin, self.xmax]
        )

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        if len(x) < 4:
            raise ValueError("At least 4 data points are required.")
        if np.allclose(np.ptp(x), 0):
            raise ValueError("x values must not all be identical.")

        self.xmin = float(np.min(x))
        self.xmax = float(np.max(x))

        kf = KFold(
            n_splits=min(self.cv_folds, len(x)),
            shuffle=True,
            random_state=42
        )

        means, stds = [], []

        for degree in range(self.max_degree + 1):
            scores = []

            for train, test in kf.split(x):
                # Avoid degrees that are too large for a training fold.
                if degree >= len(train):
                    scores.append(np.nan)
                    continue

                model = self._fit_degree(
                    x[train], y[train], degree
                )
                pred = model(x[test])

                if len(test) < 2:
                    scores.append(np.nan)
                else:
                    scores.append(r2_score(y[test], pred))

            arr = np.asarray(scores, dtype=float)
            means.append(np.nanmean(arr))
            stds.append(np.nanstd(arr, ddof=1))

        self.cv_mean = np.asarray(means, dtype=float)
        self.cv_std = np.asarray(stds, dtype=float)

        finite = np.isfinite(self.cv_mean)
        if not np.any(finite):
            raise RuntimeError("Cross-validation failed for all tested degrees.")

        safe_scores = np.where(finite, self.cv_mean, -np.inf)
        raw_best = int(np.argmax(safe_scores))

        if self.selection == "max":
            self.best_degree = raw_best
        elif self.selection == "one_se":
            se = self.cv_std[raw_best] / np.sqrt(self.cv_folds)
            if not np.isfinite(se):
                se = 0.0
            threshold = self.cv_mean[raw_best] - se
            candidates = np.where(self.cv_mean >= threshold)[0]
            self.best_degree = int(candidates[0])
        else:
            raise ValueError("selection must be 'max' or 'one_se'.")

        self.cheb_model = self._fit_degree(
            x, y, self.best_degree
        )

        self.poly_model = self.cheb_model.convert(
            kind=Polynomial
        )

        pred = self.cheb_model(x)
        self.training_r2 = float(r2_score(y, pred))

        residuals = y - pred
        dof = max(len(x) - self.best_degree - 1, 1)
        self.residual_variance = float(
            np.sum(residuals ** 2) / dof
        )

        return self

    def predict(self, x):
        return self.cheb_model(np.asarray(x, dtype=float))

    def derivative(self, order=1):
        return self.poly_model.deriv(order)

    def polynomial_coefficients(self):
        return np.asarray(self.poly_model.coef, dtype=float)

    def roots(self):
        roots = self.poly_model.roots()
        return np.array([
            r.real for r in roots
            if abs(r.imag) < 1e-8
            and self.xmin <= r.real <= self.xmax
        ], dtype=float)

    def extrema(self):
        d1 = self.poly_model.deriv(1)
        d2 = self.poly_model.deriv(2)
        result = []

        for root in d1.roots():
            if abs(root.imag) < 1e-8 and self.xmin <= root.real <= self.xmax:
                x = float(root.real)
                second = float(d2(x))

                if second > 0:
                    kind = "Minimum"
                elif second < 0:
                    kind = "Maximum"
                else:
                    kind = "Degenerate"

                result.append({
                    "x": x,
                    "y": float(self.poly_model(x)),
                    "type": kind
                })

        return result

    def inflection_points(self):
        d2 = self.poly_model.deriv(2)
        d3 = self.poly_model.deriv(3)
        result = []

        for root in d2.roots():
            if abs(root.imag) < 1e-8 and self.xmin <= root.real <= self.xmax:
                x = float(root.real)
                # Keep roots with an actual curvature sign change when possible.
                eps = max((self.xmax - self.xmin) * 1e-6, 1e-9)
                xl = max(self.xmin, x - eps)
                xr = min(self.xmax, x + eps)
                sign_change = np.sign(d2(xl)) != np.sign(d2(xr))
                if sign_change or abs(d3(x)) > 1e-10:
                    result.append({
                        "x": x,
                        "y": float(self.poly_model(x))
                    })

        return result

    def local_exponent(self, x):
        x = np.asarray(x, dtype=float)
        y = self.poly_model(x)
        dy = self.poly_model.deriv(1)(x)

        with np.errstate(divide="ignore", invalid="ignore"):
            exponent = x * dy / y

        return exponent


# ----------------------------
# Candidate physical models
# ----------------------------

def linear(x, a, b):
    return a + b * x

def quadratic(x, a, b, c):
    return a + b * x + c * x**2

def cubic(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3

def exponential(x, A, k, C):
    return A * np.exp(k * x) + C

def exponential_decay(x, A, tau, C):
    return A * np.exp(-x / tau) + C

def power_law(x, A, n, C):
    return A * np.power(x, n) + C

def inverse_law(x, A, B, C):
    return A / (x + B) + C

def logarithmic(x, A, B, C):
    return A * np.log(x + B) + C

def saturation(x, A, K, C):
    return A * x / (K + x) + C


class PhysicalModelDiscovery:
    def __init__(self, cv_folds=5):
        self.cv_folds = int(cv_folds)
        self.results = {}

        self.models = {
            "Linear": {
                "func": linear,
                "p0": None,
                "param_names": ["a", "b"]
            },
            "Quadratic": {
                "func": quadratic,
                "p0": None,
                "param_names": ["a", "b", "c"]
            },
            "Cubic": {
                "func": cubic,
                "p0": None,
                "param_names": ["a", "b", "c", "d"]
            },
            "Exponential": {
                "func": exponential,
                "p0": [1.0, 0.1, 0.0],
                "param_names": ["A", "k", "C"]
            },
            "Exponential decay": {
                "func": exponential_decay,
                "p0": [1.0, 1.0, 0.0],
                "param_names": ["A", "tau", "C"]
            },
            "Power law": {
                "func": power_law,
                "p0": [1.0, 1.0, 0.0],
                "param_names": ["A", "n", "C"]
            },
            "Inverse": {
                "func": inverse_law,
                "p0": [1.0, 1.0, 0.0],
                "param_names": ["A", "B", "C"]
            },
            "Logarithmic": {
                "func": logarithmic,
                "p0": [1.0, 1.0, 0.0],
                "param_names": ["A", "B", "C"]
            },
            "Saturation": {
                "func": saturation,
                "p0": [1.0, 1.0, 0.0],
                "param_names": ["A", "K", "C"]
            }
        }

    def _fit(self, func, x, y, p0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return curve_fit(
                func, x, y, p0=p0,
                maxfev=100000
            )

    def _aicc(self, residuals, k):
        n = len(residuals)
        rss = float(np.sum(residuals ** 2))
        rss = max(rss, 1e-300)

        aic = n * np.log(rss / n) + 2 * k

        if n - k - 1 <= 0:
            return np.inf

        return float(
            aic + 2 * k * (k + 1) / (n - k - 1)
        )

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.results = {}

        for name, info in self.models.items():
            func = info["func"]
            p0 = info["p0"]
            param_names = info["param_names"]

            # Domain guards.
            if name == "Power law" and np.any(x <= 0):
                self.results[name] = {
                    "failed": True,
                    "reason": "Power-law model requires x > 0 in this implementation."
                }
                continue

            try:
                params, covariance = self._fit(
                    func, x, y, p0
                )

                prediction = func(x, *params)
                if not np.all(np.isfinite(prediction)):
                    raise RuntimeError("Model produced non-finite predictions.")

                residuals = y - prediction
                errors = np.sqrt(
                    np.maximum(np.diag(covariance), 0)
                )
                k = len(params)

                kf = KFold(
                    n_splits=min(self.cv_folds, len(x)),
                    shuffle=True,
                    random_state=42
                )

                cv_scores = []
                cv_rmse = []

                for train, test in kf.split(x):
                    p, _ = self._fit(
                        func,
                        x[train],
                        y[train],
                        p0
                    )

                    pred = func(x[test], *p)
                    if not np.all(np.isfinite(pred)):
                        raise RuntimeError(
                            "Non-finite CV prediction."
                        )

                    if len(test) >= 2:
                        cv_scores.append(
                            r2_score(y[test], pred)
                        )

                    cv_rmse.append(
                        np.sqrt(
                            np.mean((y[test] - pred) ** 2)
                        )
                    )

                self.results[name] = {
                    "func": func,
                    "params": np.asarray(params, dtype=float),
                    "errors": np.asarray(errors, dtype=float),
                    "param_names": param_names,
                    "training_r2": float(r2_score(y, prediction)),
                    "cv_r2": float(np.mean(cv_scores)) if cv_scores else np.nan,
                    "cv_rmse": float(np.mean(cv_rmse)),
                    "aicc": self._aicc(residuals, k)
                }

            except Exception as exc:
                self.results[name] = {
                    "failed": True,
                    "reason": str(exc)
                }

        return self
