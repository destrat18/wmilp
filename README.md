# WMI-LP: LP-Based Weighted Model Integration over Non-Linear Real Arithmetic

This file provides instructions for installing and using WMI-LP, as introduced in the paper "LP-Based Weighted Model Integration over Non-Linear Real Arithmetic," and for running experiments. Tested on Ubuntu 20.04 and Python 3.8.10.

## WMI-LP

### Installation
To install WMI-LP, follow these steps:

1. Navigate to the project directory:
    ```
    cd wmi-lp
    ```
2. Install the package:
    ```
    pip install .
    ```

WMI-LP is primarily developed using the following packages:
- [sympy](https://www.sympy.org/en/index.html): Symbolic mathematics
- [gurobipy](https://www.gurobi.com/): Linear programming solver

For the complete list of dependencies, see the `requirements.txt` file.

#### Gurobi License Environment Variable [Optional]
Set the `GRB_LICENSE_FILE` environment variable to the path of your Gurobi license file. Without this, Gurobi will run in a limited capacity.

### Usage
To use WMI-LP, follow these steps:

1. Run the WMI-LP command:
    ```
    wmilp -i WMI-LP/examples/one_over_x.json --max-workers 16 --epsilon 0.1
    ```

    - `-i`: Specifies the input file.
    - `--max-workers`: Sets the maximum number of worker threads.
    - `--epsilon`: Defines the precision parameter.

### CLI Example
Use WMI-LP to compute the weighted model integration of `1/x` over `[0.1, 1]` with an epsilon of `0.1`:
```
wmilp -i WMI-LP/examples/one_over_x.json --max-workers 16 --epsilon 0.1
```

The input file `examples/one_over_x.json` is as follows:
```json
{
    "w": "1/x", 
    "bounds": [[0.1, 1]],
    "S": 1,
    "variables": ["x"]
}
```

In the input file:
- `w` is the weight function.
- `bounds` are the intervals for each variable.
- `variables` lists the variables involved.
- `S` is the semi-algebraic set.

For a detailed explanation of the input file, please refer to Chapter 2: Preliminaries.

## Experiments

### Installing Other Tools
To run experiments, you need to install the following tools:
- [GuBPi](https://github.com/gubpi-tool/gubpi)
- [WMI-PA With Volesti and Latte](https://github.com/unitn-sml/wmi-pa)
- [PSI Solver](https://github.com/eth-sri/psi)

To make it more convenient, we provide a Dockerfile that includes the installation process for all the tools.

To build the Docker image (see [Docker documentation](https://docs.docker.com/engine/install/)), run the following command:

Set the directory to the root of the project, which contains this README and the Dockerfile. Then:
```
docker build -t wmilp .
docker run -it wmilp
```

The image creation process can take 5-10 minutes.

### Dataset
The dataset used in the experiments is located in the `data` directory. It contains JSON files with random benchmarks for different templates.

#### Generating Benchmarks
To generate random benchmarks, we provide a script called `benchmark` which generates coefficients and constants for each of the four templates.

### Experimental Results

The experimental results are stored in the `paper_results` directory. Each CSV file contains details such as:
- **Benchmark**: The mathematical formula used.
- **Bounds**: Integration intervals.
- **Index**: Benchmark identifier.
- **Output**: Computed integration results.
- **Error**: Integration error bounds.
- **Time**: Execution time.
- **Details**: Additional computation information.

#### Running Experiments on the Sample Benchmark
To run experiments, follow these steps:

1. Navigate to the `experiments` directory:
    ```
    cd experiments
    ```

2. Run the experiments using different tools:

#### WMI-LP
Run the experiment with WMI-LP:
```
python3 experiments.py -b rational -p data/random_benchmarks_rational_sample.json --wmilp --max-workers 4 --epsilon 0.1
```

`epsilon` indicates the integration error bound, and `max-workers` specifies the number of computation threads. The progress displays the current error, integration region volume, and the number of checked hyper-rectangles.

Expected result:
```
INFO:root:Upper bound for weight function found: [0, 7.284371619412961]
INFO:root:Starting integral calculation with inputs: [-8.67*x**3*y_edc0fd37 - 0.43*x**2*y_edc0fd37 - 7.22*x*y_edc0fd37 + 58.6272*x - 9.18*y_edc0fd37 + 65.5712], bounds: [[0.1, 1], [0, 7.284371619412961]], and Handelman degrees: [4]
INFO:root:#HyperR Checked: 250, Error: 0.252890, Volume: (5.656414,5.909304), Time: 58.71s
INFO:root:#HyperR Checked: 500, Error: 0.129646, Volume: (5.727640,5.857286), Time: 92.18s
INFO:root:#HyperR Checked: 741, Error: 0.099635, Volume: (5.740044,5.839679), Time: 124.23s
INFO:root:Result: (5.740043932643498,5.839679411153656)
INFO:root:Bench 0 (9.92*((5.91*x**1+6.61*x**0) / (8.67*x**3 + 0.43*x**2 + 7.22*x**1 + 9.18*x**0))) is done: (5.740043932643498, 5.839679411153656)
```

#### WMI-PA with Volesti
Run the experiment with Volesti:
```
python3 experiments.py -b rational -p data/random_benchmarks_rational_sample.json --volesti
```
Expected result:
```
INFO:root:Bench 0 (9.92*((5.91*x**1+6.61*x**0) / (8.67*x**3 + 0.43*x**2 + 7.22*x**1 + 9.18*x**0))) is done: (5.34639, 6.16425)
```

#### WMI-PA with Latte
Run the experiment with Latte:
```
python3 experiments.py -b rational -p data/random_benchmarks_rational_sample.json --latte
```
Expected result:
```
INFO:root:Bench 0 (9.92*((5.91*x**1+6.61*x**0) / (8.67*x**3 + 0.43*x**2 + 7.22*x**1 + 9.18*x**0))) is failed: Not a monomial: (5818516255047572772725882617856.0 + (272544879049069317099666538496.0 * (x ^ 2.0)) + (4576218666823907981777111613440.0 * x) + (5495265351989374410452212121600.0 * (x ^ 3.0)))
```

#### GuBPi
Run the experiment with Gurobi:
```
python3 experiments.py -b rational -p data/random_benchmarks_rational_sample.json --gubpi
```
Expected result:
```
INFO:root:Bench 0 (9.92*((5.91*x**1+6.61*x**0) / (8.67*x**3 + 0.43*x**2 + 7.22*x**1 + 9.18*x**0))) is done: (0.0, 5.787299548244)
```

#### PSI Solver
Run the experiment with PSI:
```
python3 experiments.py -b rational -p data/random_benchmarks_rational_sample.json --psi
```
Expected result:
```
INFO:root:Bench 0 (9.92*((5.91*x**1+6.61*x**0) / (8.67*x**3 + 0.43*x**2 + 7.22*x**1 + 9.18*x**0))) is done: E[p_] = 10/9*Integrate[(36642/625*xi1+40982/625)*1/(361/50*xi1+43/100*xi1^2+459/50+867/100*xi1^3)*Boole[-1+xi1<=0]*Boole[-xi1+1/10<=0]*Boole[43*xi1^2+722*xi1+867*xi1^3+918!=0],{xi1,-Infinity,Infinity}]
```
