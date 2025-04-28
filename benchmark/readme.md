## Automatic DL parallel optimization
TransPred can be used to optimize distributed DL training using the Alpa compiler.
 In order to run the optimization based on TransPred's scalable and transferrable prediction use the following command.
```bash
cd benchmark
python run_benchmark.py --model=<model_name>
```

The option, `--model` specifies which benchmark to run. There are 2 benchmarks available. You can set it to either `moe` or `gpt` based on which benchmark you want to run.

To run the benchmark with latency data from a different platform to perform even faster optimization, use the following command.
```bash
python run_benchmark.py --model=<model_name> --history_path=<path_to_the_file>
```