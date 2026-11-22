from __future__ import annotations

import csv

LIBRARIES = (("torch", "PyTorch"), ("tensorflow", "TensorFlow"), ("jax", "JAX"),
             ("keras", "Keras"), ("paddle", "PaddlePaddle"), ("oneflow", "OneFlow"),
             ("mindspore", "MindSpore"), ("chainer", "Chainer"), ("numpy", "NumPy"),
             ("scipy", "SciPy"), ("sklearn", "scikit-learn"), ("opencv", "OpenCV"))
COLUMNS = ("Crash", "NaN", "Reported", "Confirmed", "Fixed", "Pending", "Excluded")


def summarize(path):
    counts = {key: {column: 0 for column in COLUMNS} for key, _ in LIBRARIES}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            values = counts[row["Library"].strip()]
            values["Reported"] += 1
            oracle = row["Bug Type"].strip()
            if oracle not in ("Crash/Exception", "NaN/Inf"):
                raise ValueError(f"unknown bug type: {oracle}")
            values["Crash" if oracle == "Crash/Exception" else "NaN"] += 1
            status = row["Bug Status"].strip()
            if status in ("Confirmed", "Fixed"):
                values["Confirmed"] += 1
                values["Fixed"] += int(status == "Fixed")
            elif status in ("Pending", "Others"):
                values["Pending" if status == "Pending" else "Excluded"] += 1
            else:
                raise ValueError(f"unknown bug status: {status}")
    rows = [{"Library": name, **counts[key]} for key, name in LIBRARIES]
    rows.append({"Library": "Total", **{column: sum(row[column] for row in rows) for column in COLUMNS}})
    return rows


def markdown(rows):
    columns = ("Library", *COLUMNS)
    lines = ["| " + " | ".join(columns) + " |", "|---|" + "---:|" * len(COLUMNS)]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    return "\n".join(lines)
