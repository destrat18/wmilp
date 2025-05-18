
from wmipa import WMI
from wmipa.integration import LatteIntegrator, VolestiIntegrator
import pandas as pd
import time, argparse, os, logging
import sympy as sym
import numpy as np
from wmilp import calculate_approximate_volume, calculate_approximate_wmi
import benchmarks as ExperimentsBenchmarks
from subprocess import check_output
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import pandas as pd
import signal
import wolframalpha
import asyncio
from pprint import  pprint


def OutOfTimeHandler(signum, frame):
    raise Exception('Timeout')


def evaluate_psi(
    benchmarks,
    result_dir,
    benchmark_name,
    timeout
):
        results = []    
        
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_psi_{int(time.time())}.csv")

        for _, bench in enumerate(benchmarks):
            bench_i = bench['index']
            start_time = time.time()
            try:
                
                if "psi" not in  bench:
                    raise Exception('Missing input formula')
                if bench['psi']['formula'] is None:
                    raise Exception('N\S')
                
                program_path = os.path.join(result_dir, f"psi_template_bench_{benchmark_name}_{bench_i}.psi")
                
                with open(program_path, 'w') as f:
                        if len(bench['wmilp']['bounds']) == 1:
                            f.write(
                                ExperimentsBenchmarks.PSI_SOLVER_ONE_VAR_TEMPLATE.format(
                                    x_lower_bound = bench['wmilp']['bounds'][0][0],
                                    x_upper_bound = bench['wmilp']['bounds'][0][1],
                                    formula = bench['psi']['formula']
                                )
                            )
                        elif len(bench['wmilp']['bounds']) == 2:
                            f.write(
                                ExperimentsBenchmarks.PSI_SOLVER_TWO_VAR_TEMPLATE.format(
                                    x_lower_bound = bench['wmilp']['bounds'][0][0],
                                    x_upper_bound = bench['wmilp']['bounds'][0][1],
                                    y_lower_bound = bench['wmilp']['bounds'][1][0],
                                    y_upper_bound = bench['wmilp']['bounds'][1][1],
                                    formula = bench['psi']['formula']
                                )
                            )
                        elif len(bench['wmilp']['bounds']) == 3:
                            f.write(
                                ExperimentsBenchmarks.PSI_SOLVER_THREE_VAR_TEMPLATE.format(
                                    x_lower_bound = bench['wmilp']['bounds'][0][0],
                                    x_upper_bound = bench['wmilp']['bounds'][0][1],
                                    y_lower_bound = bench['wmilp']['bounds'][1][0],
                                    y_upper_bound = bench['wmilp']['bounds'][1][1],
                                    z_lower_bound = bench['wmilp']['bounds'][2][0],
                                    z_upper_bound = bench['wmilp']['bounds'][2][1],
                                    formula = bench['psi']['formula']
                                )
                            )
                
                cmd = []
                if timeout is not None:
                    cmd = ["timeout", str(int(timeout))]
                cmd += ['psi', program_path, '--expectation', '--mathematica']
                
                output = check_output(cmd).decode("utf-8").strip().replace('\n', '\t')
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['wmilp']['w'],
                    "bounds": bench['wmilp']['bounds'],
                    "index": bench_i,
                    'output': output,
                    'error': None,
                    "time": time.time()-start_time,
                    'details': []
                })
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {output}")
            
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['wmilp']['w'],
                    "bounds": bench['wmilp']['bounds'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })      
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)
            
def evaluate_volesi(
    benchmarks,
    benchmark_name,
    result_dir,
    repeat         
        
):

    mode = WMI.MODE_SAE4WMI
    
    results = []    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_volesti_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):
        
        bench_i = bench['index']
        start_time = time.time()
        try:
                details = []
                if "wmipa" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmipa']['w'] is None:
                    raise Exception('N\S')
        
                for N in [1000]:
                    for i in range(args.repeat):

                        integrator = VolestiIntegrator(N=N) 
                        start_time = time.time()
                        wmi = WMI(bench['wmipa']['chi'], bench['wmipa']['w'], integrator=integrator)
                        volume, n_integrations = wmi.computeWMI(bench['wmipa']['phi'], mode=mode)

                        details.append(
                            {
                                'N': N,
                                'repeat': i,
                                'n_integrations': n_integrations,
                                'mode': mode,
                                'output': volume,
                                'time': time.time()-start_time
                            }
                        )
                
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['wmilp']['w'],
                        "bounds": bench['wmilp']['bounds'],
                        "index": bench_i,
                        'output': (min([d['output'] for d in details]), max([d['output'] for d in details])),
                        'error': None,
                        "time": (min([d['time'] for d in details]), max([d['time'] for d in details])),
                        'details': details
                    })
        
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
                       

        except Exception as e:
            logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            }) 
        
        
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)


def evaluate_gubpi(
    benchmarks,
    result_dir,
    benchmark_name
):
    results = []
    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_gubpi_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):
        bench_i = bench['index']
        start_time = time.time()
        try:
            if "gubpi" not in  bench:
                raise Exception('Missing input formula')
            if bench['gubpi']['formula'] is None:
                raise Exception('N\S')
            
            
            program_path = os.path.join(result_dir, f"gubpi_template_bench_{benchmark_name}_{bench_i}.spcf")
            
            with open(program_path, 'w') as f:
                    if len(bench['wmilp']['bounds']) == 1:
                        f.write(
                            ExperimentsBenchmarks.GUBPI_SOLVER_ONE_VAR_TEMPLATE.format(
                                x_lower_bound = bench['wmilp']['bounds'][0][0],
                                x_upper_bound = bench['wmilp']['bounds'][0][1],
                                formula = bench['gubpi']['formula']
                            )
                        )
                    elif len(bench['wmilp']['bounds']) == 2:
                        f.write(
                            ExperimentsBenchmarks.GUBPI_SOLVER_TWO_VAR_TEMPLATE.format(
                                x_lower_bound = bench['wmilp']['bounds'][0][0],
                                x_upper_bound = bench['wmilp']['bounds'][0][1],
                                y_lower_bound = bench['wmilp']['bounds'][1][0],
                                y_upper_bound = bench['wmilp']['bounds'][1][1],
                                formula = bench['gubpi']['formula']
                            )
                        )
                    elif len(bench['wmilp']['bounds']) == 3:
                        f.write(
                            ExperimentsBenchmarks.GUBPI_SOLVER_THREE_VAR_TEMPLATE.format(
                                x_lower_bound = bench['wmilp']['bounds'][0][0],
                                x_upper_bound = bench['wmilp']['bounds'][0][1],
                                y_lower_bound = bench['wmilp']['bounds'][1][0],
                                y_upper_bound = bench['wmilp']['bounds'][1][1],
                                z_lower_bound = bench['wmilp']['bounds'][2][0],
                                z_upper_bound = bench['wmilp']['bounds'][2][1],
                                formula = bench['gubpi']['formula']
                            )
                        )
            
            raw_output = check_output(['GuBPI', program_path]).decode("utf-8")
            output = (
                float(raw_output.split("\n")[-6].split(" ")[2]),
                float(raw_output.split("\n")[-6].split(" ")[-1])
                )
            
            # Remove normalization
            norm = np.prod([b[1]-b[0] for b in bench['wmilp']['bounds']])
            print(norm)
            output = (output[0]*norm, output[1]*norm)    
            
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                'output': output,
                'error': None,
                "time": time.time()-start_time,
                'details': {
                    "raw_output": raw_output    
                }
            })
            logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {output}")
            
        except Exception as e:
            logging.exception(e)
            logging.error(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            })      
            
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)
        
        
        
        
def evaluate_latte(
    benchmarks,
    result_dir,
    benchmark_name        
):


    mode = WMI.MODE_SAE4WMI
    
    results = []    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_latte_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):
        
        bench_i = bench['index']
        start_time = time.time()
        try:
                
                if "wmipa" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmipa']['w'] is None:
                    raise Exception('N\S')


                integrator = LatteIntegrator() 
                start_time = time.time()
                wmi = WMI(bench['wmipa']['chi'], bench['wmipa']['w'], integrator=integrator)
                volume, n_integrations = wmi.computeWMI(bench['wmipa']['phi'], mode=mode)
               
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['wmilp']['w'],
                        "bounds": bench['wmilp']['bounds'],
                        "index": bench_i,
                        'output': volume,
                        'error': None,
                        "time": time.time()-start_time,
                        'details': {
                                'n_integrations': n_integrations,
                                'mode': mode,
                                'output': volume,
                }
                    })
        
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
                       

        except Exception as e:
            logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            }) 
        
        
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)


def evaluate_wmilp(
        benchmarks,
        result_dir,
        epsilon,
        max_workers,
        benchmark_name,
        timeout    
        
):


        mode = WMI.MODE_SAE4WMI

        results = []
        
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_wmilp_{int(time.time())}.csv")

        for _, bench in enumerate(benchmarks):

            bench_i = bench['index']
            # integrator = FazaIntegrator(threshold=epsilon, max_workers=max_workers) 
            try:
                
                if "wmilp" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmilp']['w'] is None:
                    raise Exception('N\S')
                
                start_time = time.time()
                if timeout is not None:
                    signal.alarm(int(timeout))

                output = calculate_approximate_wmi(
                        S=bench['wmilp']['S'],
                        bounds=bench['wmilp']['bounds'],
                        max_workers=max_workers,
                        epsilon=epsilon,
                        w=bench['wmilp']['w'],
                        variables=bench['wmilp']['variables']
                )
                
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['wmilp']['w'],
                        "bounds": bench['wmilp']['bounds'],
                        "index": bench_i,
                        'output': (output[0], output[1]),
                        'error': None,
                        "time": time.time()-start_time,
                        'details': {
                                'n_integrations': 1,
                                'mode': mode,
                                'output': output,
                    }
                })
                
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
                print(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
                
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
        
                # if len(integrator.logs)==1:
                #     output = integrator.logs[0]['volume']
                # else:
                #     output = None
                    
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['wmilp']['w'],
                    "bounds": bench['wmilp']['bounds'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })
                logging.exception(e)
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False) 
                        
                        
def evaluate_mathematica(
        benchmarks,
        result_dir,
        benchmark_name,
        timeout,
        repeat    
        
):
    
    
    with WolframLanguageSession() as session:

        # if session.is_alive() == False:
        #     return

        mode = WMI.MODE_SAE4WMI

        results = []
        
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_mathematica_{int(time.time())}.csv")

        for _, bench in enumerate(benchmarks):

            bench_i = bench['index']
            # integrator = FazaIntegrator(threshold=epsilon, max_workers=max_workers) 
            try:
                
                if "wmilp" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmilp']['w'] is None:
                    raise Exception('N\S')
                
                start_time = time.time()
                
                if timeout is not None:
                    signal.alarm(int(timeout))
                
                output = session.evaluate(bench['mathematica']['formula'], timeout=timeout)
                
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['wmilp']['w'],
                        "bounds": bench['wmilp']['bounds'],
                        "index": bench_i,
                        'output': output,
                        'error': None,
                        "time": time.time()-start_time,
                        'details': {}
                })
                
                logging.info(f"Bench {bench_i} ({bench['mathematica']['formula']}) is done: {results[-1]['output']}")
                # print(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
                
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
        
                # if len(integrator.logs)==1:
                #     output = integrator.logs[0]['volume']
                # else:
                #     output = None
                    
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['wmilp']['w'],
                    "bounds": bench['wmilp']['bounds'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })
                logging.exception(e)
            time.sleep(1)
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)

                        
def evaluate_mathematica_nested(
        benchmarks,
        result_dir,
        benchmark_name,
        timeout,
        repeat    
        
):
    
    
    with WolframLanguageSession() as session:

        # if session.is_alive() == False:
        #     return

        mode = WMI.MODE_SAE4WMI

        results = []
        
        result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_mathematica_nested_{int(time.time())}.csv")

        for _, bench in enumerate(benchmarks):

            bench_i = bench['index']
            # integrator = FazaIntegrator(threshold=epsilon, max_workers=max_workers) 
            try:
                
                if "wmilp" not in  bench:
                    raise Exception('Missing input formula')
                if bench['wmilp']['w'] is None:
                    raise Exception('N\S')
                
                start_time = time.time()
                
                if timeout is not None:
                    signal.alarm(int(timeout))
                
                output = session.evaluate(bench['mathematica_nested']['formula'], timeout=timeout)
                
                results.append({
                        "bechmark": benchmark_name,
                        "formula": bench['wmilp']['w'],
                        "bounds": bench['wmilp']['bounds'],
                        "index": bench_i,
                        'output': output,
                        'error': None,
                        "time": time.time()-start_time,
                        'details': {}
                })
                
                logging.info(f"Bench {bench_i} ({bench['mathematica']['formula']}) is done: {results[-1]['output']}")
                # print(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
                
            except Exception as e:
                logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
        
                # if len(integrator.logs)==1:
                #     output = integrator.logs[0]['volume']
                # else:
                #     output = None
                    
                results.append({
                    "bechmark": benchmark_name,
                    "formula": bench['wmilp']['w'],
                    "bounds": bench['wmilp']['bounds'],
                    "index": bench_i,
                    "output": None,
                    'error': str(e),
                    "time": time.time()-start_time,
                    'details': []
                })
                logging.exception(e)
            time.sleep(1)
                
            pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)

def evaluate_wolfram_alpha(
        benchmarks,
        result_dir,
        benchmark_name,
        timeout,
        wolfram_alpha_key,
        repeat    
):
    
    
    results = []
    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_wolfram_alpha_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):

        bench_i = bench['index']
        # integrator = FazaIntegrator(threshold=epsilon, max_workers=max_workers) 
        try:
            
            if "wmilp" not in  bench:
                raise Exception('Missing input formula')
            if bench['wmilp']['w'] is None:
                raise Exception('N\S')
            
            start_time = time.time()
            
            if timeout is not None:
                signal.alarm(int(timeout))
            
            
            details = []
            output = None
            
            client = wolframalpha.Client(wolfram_alpha_key)
            
            for i in range(repeat):
                try:
                    res = client.query(bench['mathematica']['formula'])
                    # pprint(res)
                    if 'pod' in res:
                        result_pods = res['pod']
                        if type(result_pods) != list:
                            result_pods = [result_pods]
                        for pod in result_pods:
                            if pod['@title'] == 'Definite integral':
                                plain_text = pod['subpod']['plaintext']
                                if plain_text:
                                    output = plain_text.split('=')[1].strip()
                                    
                                    try:
                                        output = pd.to_numeric(output)
                                    except:
                                        pass
                                    
                                break
                    
                    if output is None:
                        raise Exception('No output found')
                    
                    details.append({
                        'repeat': i,
                        'output': output,
                        'time': time.time() - start_time
                    })
                except Exception as e:
                    details.append({
                        'repeat': i,
                        'output': None,
                        'error': str(e),
                        'time': time.time() - start_time
                    })
                    
                time.sleep(1)
                    
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                'output': (min([d['output'] for d in details if isinstance(d['output'], (int, float))], default=None),
                           max([d['output'] for d in details if isinstance(d['output'], (int, float))], default=None)),
                'error': None if any(isinstance(d['output'], (int, float)) for d in details) else 'All attempts failed',
                "time": (min([d['time'] for d in details]), max([d['time'] for d in details])),
                'details': details
            })
            
            logging.info(f"Bench {bench_i} ({bench['mathematica']['formula']}) is done: {results[-1]['output']}")
            # print(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
            
        except Exception as e:
            logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
    
            # if len(integrator.logs)==1:
            #     output = integrator.logs[0]['volume']
            # else:
            #     output = None
                
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            })
            logging.exception(e)
        time.sleep(1)
            
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)

def evaluate_wolfram_alpha_nested(
        benchmarks,
        result_dir,
        benchmark_name,
        timeout,
        wolfram_alpha_key,
        repeat    
):
    
    
    results = []
    
    result_path = os.path.join(result_dir, f"benchmark_{benchmark_name}_wolfram_alpha_nested_{int(time.time())}.csv")

    for _, bench in enumerate(benchmarks):

        bench_i = bench['index']
        # integrator = FazaIntegrator(threshold=epsilon, max_workers=max_workers) 
        try:
            
            if "wmilp" not in  bench:
                raise Exception('Missing input formula')
            if bench['wmilp']['w'] is None:
                raise Exception('N\S')
            
            start_time = time.time()
            
            if timeout is not None:
                signal.alarm(int(timeout))
            
            
            details = []
            output = None
            
            client = wolframalpha.Client(wolfram_alpha_key)
            
            for i in range(repeat):
                try:
                    res = client.query(bench['mathematica_nested']['formula'])
                    # pprint(res)
                    if 'pod' in res:
                        result_pods = res['pod']
                        if type(result_pods) != list:
                            result_pods = [result_pods]
                        for pod in result_pods:
                            if pod['@title'] == 'Definite integral':
                                plain_text = pod['subpod']['plaintext']
                                if plain_text:
                                    output = plain_text.split('=')[1].strip()
                                    
                                    try:
                                        output = pd.to_numeric(output)
                                    except:
                                        pass
                                    
                                break
                    
                    if output is None:
                        raise Exception('No output found')
                    
                    details.append({
                        'repeat': i,
                        'output': output,
                        'time': time.time() - start_time
                    })
                except Exception as e:
                    details.append({
                        'repeat': i,
                        'output': None,
                        'error': str(e),
                        'time': time.time() - start_time
                    })
                    
                time.sleep(1)
                    
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                'output': (min([d['output'] for d in details if isinstance(d['output'], (int, float))], default=None),
                           max([d['output'] for d in details if isinstance(d['output'], (int, float))], default=None)),
                'error': None if any(isinstance(d['output'], (int, float)) for d in details) else 'All attempts failed',
                "time": (min([d['time'] for d in details]), max([d['time'] for d in details])),
                'details': details
            })
            
            logging.info(f"Bench {bench_i} ({bench['mathematica_nested']['formula']}) is done: {results[-1]['output']}")
            # print(f"Bench {bench_i} ({bench['wmilp']['w']}) is done: {results[-1]['output']}")
            
        except Exception as e:
            logging.info(f"Bench {bench_i} ({bench['wmilp']['w']}) is failed: {e}")
    
            # if len(integrator.logs)==1:
            #     output = integrator.logs[0]['volume']
            # else:
            #     output = None
                
            results.append({
                "bechmark": benchmark_name,
                "formula": bench['wmilp']['w'],
                "bounds": bench['wmilp']['bounds'],
                "index": bench_i,
                "output": None,
                'error': str(e),
                "time": time.time()-start_time,
                'details': []
            })
            logging.exception(e)
        time.sleep(1)
            
        pd.DataFrame(results).sort_values('index').to_csv(result_path, index=False)


if __name__ == "__main__":
        
    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGALRM, OutOfTimeHandler)
    
    
    parser = argparse.ArgumentParser(
            prog='WMI-LP',
            description='LP-Based Weighted Model Integration over Non-Linear Real Arithmetic'
            )

    parser.add_argument("--timeout", type=float, default=None, help='Timeout for each benchmark in seconds')        
    parser.add_argument("--epsilon", type=float, default=0.1, help="Threshold for WMI-LP")        
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of workers for WMI-LP")
    parser.add_argument("--repeat", type=int, default=10, help="Number of trials for Volesti integrator")
    parser.add_argument('--volesti', action='store_true', default=False, help="Run benchmarks using WMI-PA with Volesti")
    parser.add_argument('--latte', action='store_true', default=False, help="Run benchmarks using WMI-PA with Latte")
    parser.add_argument('--wmilp', action='store_true', default=False, help="Run benchmarks using WMI-LP")
    parser.add_argument('--psi', action='store_true', default=False, help="Run benchmarks using Psi solver")
    parser.add_argument('--gubpi', action='store_true', default=False, help="Run benchmarks using GuBPI solver")
    parser.add_argument('--mathematica', action='store_true', default=False, help="Run benchmarks using Mathematica")
    
    
    parser.add_argument('--mathematica-nested', action='store_true', default=False, help="Run benchmarks using Mathematica (Nested Integrals)")
    
    parser.add_argument('--wolfram-alpha', action='store_true', default=False, help="Run benchmarks using Wolfram Alpha")
    parser.add_argument('--wolfram-alpha-nested', action='store_true', default=False, help="Run benchmarks using Wolfram Alpha (Nested Integrals)")
    
    parser.add_argument('--wolfram-alpha-key', type=str, default=None, help="Wolfram Alpha API key")
    
    
    parser.add_argument('--benchmark', "-b", choices=['manual', 'rational', 'sqrt', "rational_sqrt", "rational_2"], default="manual", help="Type of benchmark to run")
    parser.add_argument('--benchmark-path', "-p", type=str, help="Path to the benchmark file")
    parser.add_argument('--result-dir', "-o", type=str, default="results", help="Directory to save the results")
    
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)


    if args.benchmark == 'manual':
        benchmarks = ExperimentsBenchmarks.selected_benchmark
        args.benchmark = "manual"
    elif args.benchmark == "rational":
        benchmarks = ExperimentsBenchmarks.load_rational_benchmarks(
            benchmak_path=args.benchmark_path
        )
    elif args.benchmark == "sqrt":
        benchmarks = ExperimentsBenchmarks.load_sqrt_benchmarks(
            benchmak_path=args.benchmark_path
        )
    elif args.benchmark == "rational_sqrt":
        benchmarks = ExperimentsBenchmarks.load_rational_sqrt_benchmarks(
            benchmak_path=args.benchmark_path
        )
    elif args.benchmark == "rational_2":
        benchmarks = ExperimentsBenchmarks.load_rational_2_benchmarks(
            benchmak_path=args.benchmark_path
        )
    else:
        raise NotImplementedError()

    if args.volesti:
            evaluate_volesi(
                benchmarks=benchmarks,
                benchmark_name=args.benchmark,
                result_dir=args.result_dir,
                repeat=args.repeat
            )
    if args.latte:
            evaluate_latte(
                benchmarks=benchmarks,
                result_dir=args.result_dir,
                benchmark_name=args.benchmark,

            )
            
    if args.wmilp:
            evaluate_wmilp(
                benchmarks=benchmarks,
                result_dir=args.result_dir,
                epsilon=args.epsilon,
                max_workers=args.max_workers,
                benchmark_name=args.benchmark,
                timeout=args.timeout
            )
            
    if args.mathematica:
        evaluate_mathematica(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark,
            timeout=args.timeout,
            repeat=args.repeat
        )

    if args.mathematica_nested:
        evaluate_mathematica_nested(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark,
            timeout=args.timeout,
            repeat=args.repeat
        )

    if args.wolfram_alpha:
        evaluate_wolfram_alpha(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark,
            timeout=args.timeout,
            wolfram_alpha_key=args.wolfram_alpha_key,
            repeat=args.repeat
        )
           
    if args.psi:
        evaluate_psi(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark,
            timeout=args.timeout
        )
    
    if args.gubpi:
        evaluate_gubpi(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark
        )
        
    if args.wolfram_alpha_nested:
        evaluate_wolfram_alpha_nested(
            benchmarks=benchmarks,
            result_dir=args.result_dir,
            benchmark_name=args.benchmark,
            timeout=args.timeout,
            wolfram_alpha_key=args.wolfram_alpha_key,
            repeat=args.repeat
        )