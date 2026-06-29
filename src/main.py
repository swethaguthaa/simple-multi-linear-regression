import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def build_data(rows=120, seed=4):
    rng = np.random.default_rng(seed)
    rd_spend = rng.normal(85000, 35000, rows).clip(5000, 180000)
    administration = rng.normal(55000, 18000, rows).clip(12000, 120000)
    marketing = rng.normal(70000, 30000, rows).clip(5000, 160000)
    profit = 25000 + 0.78 * rd_spend + 0.18 * marketing + 0.05 * administration + rng.normal(0, 9000, rows)
    return pd.DataFrame(
        {
            "rd_spend": rd_spend,
            "administration": administration,
            "marketing_spend": marketing,
            "profit": profit,
        }
    )


def main():
    data = build_data()
    x = data.drop(columns="profit")
    y = data["profit"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R2 score: {r2_score(y_test, predictions):.3f}")
    for feature, coefficient in zip(x.columns, model.coef_):
        print(f"{feature}: {coefficient:.3f}")


if __name__ == "__main__":
    main()

