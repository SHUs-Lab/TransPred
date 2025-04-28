### TransPred

This repository implements TransPred, a scalable automatic parallelization search for distributed DL model training. We acheive this by using few-shot learning to translate the latency of DL stages between different configurations, which allows small number of profiling to generate the parallel plan for DL model training.


#### Installation

1. TransPred is based on the [Alpa](https://https://github.com/alpa-projects/alpa) compiler for DL training. Before beginning, install Alpa following the [instructions](https://alpa.ai/install.html).

2. Clone the source code of TransPred from this repository.
    ```bash
    git clone https://github.com/SHUs-Lab/TransPred.git
    ```

3. Install TransPred in your python environment
    ```bash
    cd TransPred
    pip install -e .
    pip install -r requirements.txt
    ```

4. See the `Benchmarks` folder for examples on running the optimizer with TransPred.

