"""Verify dependencies and run a TensorFlow CPU forward pass."""
import json
import platform
import sys
from pathlib import Path


def main():
    print(f"Python {platform.python_version()} | {platform.platform()} | {sys.executable}")
    import matplotlib
    import numpy
    import openpyxl
    import pandas
    import scipy
    import sklearn
    import tensorflow as tf
    import yaml

    packages = (numpy, pandas, scipy, sklearn, matplotlib, openpyxl, yaml, tf)
    versions = {module.__name__: module.__version__ for module in packages}
    with tf.device("/CPU:0"):
        conv = tf.keras.layers.Conv1D(
            filters=1, kernel_size=3, use_bias=False, kernel_initializer="ones"
        )
        output = conv(tf.ones((1, 100, 6)))
        tf.debugging.assert_equal(tf.shape(output), [1, 98, 1])
        tf.debugging.assert_near(output, tf.fill((1, 98, 1), 18.0))

    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
        "tensorflow_cpu_forward_pass": "PASS",
        "conv1d_output_shape": output.shape.as_list(),
    }
    path = Path("results/phase1/environment.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: TensorFlow {tf.__version__}, Conv1D output {tuple(output.shape)}")
    print(path)


if __name__ == "__main__":
    main()
