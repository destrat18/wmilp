import  gurobipy as gp 
import numpy as np
from itertools import product
import time
import copy
import logging
import pandas as pd
import uuid
import queue
import threading
import argparse
logging.basicConfig(level=logging.INFO)
    

import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

import  gurobipy as gp 
import numpy as np
from itertools import product
import time
import copy
import logging
import pandas as pd
import uuid
import queue
import threading
import argparse
import multiprocessing as mp


import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

def generate_monoids(variables, d):

    # Generate all combinations of variables of degree up to d
    monoids = set()

    combinations = product(variables, repeat=d)
    for combo in combinations:
        monoids.add(sym.prod(combo))
    return list(monoids)

def generate_monoids_up_to_degree(var_list, d):

    # Generate all combinations of variables of degree up to d
    monoids = set()
    for i in range(0, d+1):
        combinations = product(var_list, repeat=i)
        for combo in combinations:
            monoids.add(sym.prod(combo))
            
    monoids = list(set(monoids))
    return list(monoids)


def generate_handelman_equations(degree, f_list, g, variables):
    
    start_time = time.time()
    # Generate all possible monoids
    monoids = generate_monoids_up_to_degree(f_list, degree)
    logging.debug(f"Monoids generation time: {time.time()-start_time:.2f}s")
    
    # create temp variable for each monoind
    temp_vars = []
    for i in range(len(monoids)):
        l = sym.symbols(f"l_{i}")
        monoids[i] = monoids[i]*l
        temp_vars.append(l)
    
    a_pos = sym.simplify(sym.expand(sum(monoids)-(g)))

    pos_equations = []
    for d in range(degree, 0, -1):
        for mn in generate_monoids(variables, d):
            new_assert = a_pos.coeff(mn)
            pos_equations.append(new_assert)
            a_pos = sym.simplify(a_pos - (new_assert)*mn)
    pos_equations.append(a_pos)

    a_neg = sym.simplify(sym.expand(sum(monoids)-(-1*g)))

    neq_equations = []
    for d in range(degree, 0, -1):
        for mn in generate_monoids(variables, d):
            new_assert = a_neg.coeff(mn)
            neq_equations.append(new_assert)
            a_neg = sym.simplify(a_neg - (new_assert)*mn)
    neq_equations.append(a_neg)

    
    return pos_equations, neq_equations, temp_vars


def generate_handelman_equations_2(degree, f_list, g, variables):
    
    start_time = time.time()
    # Generate all possible monoids
    monoids = generate_monoids_up_to_degree(f_list, degree)
    logging.debug(f"Monoids generation time: {time.time()-start_time:.2f}s")
    
    # create temp variable for each monoind
    temp_vars = []
    for i in range(len(monoids)):
        l = sym.symbols(f"l_{i}")
        monoids[i] = monoids[i]*l
        temp_vars.append(l)
    
    a_pos = sym.simplify(sym.expand(sum(monoids)))

    pos_equations = []
    for d in range(degree, 0, -1):
        for mn in generate_monoids(variables, d):
            new_assert = a_pos.coeff(mn)
            pos_equations.append(new_assert)
            a_pos = sym.simplify(a_pos - (new_assert)*mn)
    pos_equations.append(a_pos)

    g_equations = []
    for d in range(degree, 0, -1):
        for mn in generate_monoids(variables, d):
            new_assert = g.coeff(mn)
            g_equations.append(new_assert)
            g = sym.simplify(g - (new_assert)*mn)
    g_equations.append(g)

    
    return pos_equations, g_equations, temp_vars


def is_feasible_g(equations, g_constants, temp_vars):
    
    coeff_matrix = []
    for a in equations:
        coeff = []
        for v in [1]+temp_vars:
            try:
                coeff.append(float(a.coeff_monomial(v)))
            except:
                coeff.append(0)
        coeff_matrix.append(coeff)

    M = np.array(coeff_matrix)
    RHS = (M[:,:1]*-1).flatten()
    M = M[:,1:]
    
    RHS = RHS + g_constants
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            m.setObjective(True, gp.GRB.MAXIMIZE)
            
            lp_l = m.addMVar(shape=len(temp_vars), name="l", ub=float('inf'), lb=0)            
            m.addConstr( M @ lp_l == RHS, name="c")

            m.optimize()
    
            return not(m.status==gp.GRB.INFEASIBLE), m.runtime
        
# def is_feasible(equations, temp_vars):
    
#     coeff_matrix = []
#     for a in equations:
#         coeff = []
#         for v in [1]+temp_vars:
#             try:
#                 coeff.append(float(a.coeff_monomial(v)))
#             except:
#                 coeff.append(0)
#         coeff_matrix.append(coeff)

#     M = np.array(coeff_matrix)
#     RHS = (M[:,:1]*-1).flatten()
#     M = M[:,1:]
    
#     with gp.Env(empty=True) as env:
#         env.setParam('OutputFlag', 0)
#         env.start()
#         with gp.Model(env=env) as m:
#             m.setObjective(True, gp.GRB.MAXIMIZE)
            
#             lp_l = m.addMVar(shape=len(temp_vars), name="l", ub=float('inf'), lb=0)            
#             m.addConstr( M @ lp_l == RHS, name="c")

#             m.optimize()
    
#             return not(m.status==gp.GRB.INFEASIBLE), m.runtime
    

# def is_feasible_test(equations, temp_vars):
    
#     coeff_matrix = []
#     for a in equations:
#         coeff = []
#         for v in [1]+temp_vars:
#             try:
#                 coeff.append(sym.Poly(a.coeff_monomial(v)))
#             except:
#                 coeff.append(0)
#         coeff_matrix.append(coeff)

#     M = np.array(coeff_matrix)
#     RHS = (M[:,:1]*-1).flatten()
#     M = M[:,1:]
    
#     with gp.Env(empty=True) as env:
#         env.setParam('OutputFlag', 0)
#         env.start()
#         with gp.Model(env=env) as m:
#             m.setObjective(True, gp.GRB.MAXIMIZE)
            
#             lp_l = m.addMVar(shape=len(temp_vars), name="l", ub=float('inf'), lb=0)            
#             m.addConstr( M @ lp_l == RHS, name="c")

#             m.optimize()
    
#             return not(m.status==gp.GRB.INFEASIBLE), m.runtime
    

# def read_input(input_path, bounds_path):
    
#     inputs = []
#     # Read integrand and list of variables
#     with open(input_path) as f:
#         variables = sym.symbols(f.readline().strip().split(" "))
        
#         # Support language with small expressiveness
#         inp = sym.parse_expr(f.readline())
        
#         # Convert the input to CNF
#         inp = sym.to_cnf(inp)
#         if isinstance(inp, sym.core.relational.Lt) or isinstance(inp, sym.core.relational.Le):
#             inputs.append(inp.rhs - inp.lhs)
#         elif isinstance(inp, sym.core.relational.Gt) or isinstance(inp, sym.core.relational.Ge):
#             inputs.append(inp.lhs-inp.rhs)
#         else:
#             # we convert the inputs to the form g_i>0 or g_i >=0
#             for exp in inp.args:
#                 if isinstance(exp, sym.core.relational.Lt) or isinstance(exp, sym.core.relational.Le):
#                     inputs.append(exp.rhs - exp.lhs)
#                 elif isinstance(exp, sym.core.relational.Gt) or isinstance(exp, sym.core.relational.Ge):
#                     inputs.append(exp.lhs-exp.rhs)

    
#     inputs = [sym.simplify(sym.expand(i))for i in inputs]
        
        
    
#     # Read bounds
#     bounds = []
#     with open(bounds_path) as f:

#         n_ineq, n_vars = pd.to_numeric(f.readline().split(" "))
#         n_vars = n_vars -1

#         bounds = []
#         for i in range(n_vars):
#             bounds.append([None, None])

#         for i in range(n_ineq):
#             coeffs = [float(c) for c in f.readline().strip().split(" ")]
#             # print(coeffs)
#             constant = coeffs[0]
#             for j, c in enumerate(coeffs[1:]):
#                 if c < 0:
#                     bounds[j][1] = constant/abs(c)
#                 elif c > 0:
#                     bounds[j][0] = -1*constant/abs(c)    
#                 # print(j, c, bounds)
                
#     return inputs, bounds, variables


def generate_f_list(vars):
    
    f_list = []
    bound_vars = []

    for i, var in enumerate(vars):
        l = sym.Symbol(f"l_{str(var)}")
        u = sym.Symbol(f"u_{str(var)}")
        f_list.append(u-var)
        f_list.append(var-l)
        bound_vars.append([l, u])
    return f_list, bound_vars 

def apply(inp):
    return sym.Poly(inp[0].subs(inp[1]))

class Checker(mp.Process):
    
    
    command = None
    hardhat_instance = None
    
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
    
    def __init__(self, equations, equations_g, variables, to_check_queue, checked_queue):
        mp.Process.__init__(self)
        self.check_next = True
        # self.equations = equations
        self.equations = self.equations_g = equations_g
        self.variables = variables
        self.to_check_queue = to_check_queue
        self.checked_queue = checked_queue
    
    
    def run(self):
        
        # Read the queue until it finishes
        while True:
            try:
                # Get a hrect from the to_check_queue
                hrect = self.to_check_queue.get()
                if hrect == None: # Time to close the process
                    break
                
                cur_depth, cur_bounds, cur_volume, bound_vars, cur_memoization = hrect
                

                is_inside_list = []
                is_outside_list = []
                
                for equation_i, (inside_equations, outside_equations, temp_vars) in enumerate(self.equations):

                    if cur_memoization[equation_i]:
                        # it means that we already know that this hrect is inside of the current euqations
                        is_inside_list.append(True)
                        is_outside_list.append(False)
                    else:

                        # timing
                        start_time = time.time()
                        # is_feasible_test(inside_equations, temp_vars=temp_vars)
                        # Optimization porblem
                        subs_dict = {}
                        for cur_dim in range(len(cur_bounds)):
                            subs_dict[bound_vars[cur_dim][0]] = cur_bounds[cur_dim][0] 
                            subs_dict[bound_vars[cur_dim][1]] = cur_bounds[cur_dim][1]



                        
                        # with mp.Pool(16) as p:
                        #     cur_inside_equations_ = p.map(apply, [(a, subs_dict) for a in inside_equations])
                    
                        # with mp.Pool(16) as p:
                        #     cur_outside_equations_ = p.map(apply, [(a, subs_dict) for a in outside_equations])
                        
                        # cur_inside_equations_ = [
                        #     sym.Poly(a.subs(subs_dict)) for a in inside_equations
                        #     ]
                    
                        # cur_outside_equations_ = [
                        #     sym.Poly(a.subs(subs_dict)) for a in outside_equations
                        #     ]
                        
                        
                        cur_equations, cur_g_constants  = self.equations_g[equation_i][0], self.equations_g[equation_i][1]
                        cur_g_constants = np.asarray(cur_g_constants)
                        cur_equations_ = [
                            sym.Poly(a.subs(subs_dict)) for a in cur_equations
                            ]
                        
                        subs_time = time.time()-start_time


                        start_time = time.time()
                        # is_inside, inside_runtime = is_feasible(cur_inside_equations_, temp_vars)
                        is_inside, inside_runtime = is_feasible_g(cur_equations_, cur_g_constants, temp_vars)
                        # assert(is_inside==is_inside_g)
                        
                        is_outside, outside_runtime = is_feasible_g(cur_equations_, cur_g_constants*-1, temp_vars)
                        # is_outside, outside_runtime = is_feasible(cur_outside_equations_, temp_vars)
                        # assert(is_outside==is_outside_g)
                        solver_time = time.time()-start_time
                        
                        
                        stats = {
                            "solver_time": solver_time,
                            "subs_time": subs_time
                        }
                        
                        is_inside_list.append(is_inside)
                        is_outside_list.append(is_outside)
                    
                            
                # if it's inside or outside remove it from error
                if sum(is_inside_list)==len(self.equations):
                    self.checked_queue.put((
                            0, # Inside
                            cur_volume, # volume of hyper-rect
                            cur_bounds,
                            stats
                        ))
                elif sum(is_outside_list)>0:
                    self.checked_queue.put(
                        (
                            1, # mean it is outside
                            cur_volume, # volume of hyper-rect
                            cur_bounds,
                            stats
                        )
                    )
                else:
                    split_step=2
                    
                    self.checked_queue.put(
                        (
                            2, # means it must be splitted
                            cur_volume, # volume of hyper-rect
                            cur_bounds,
                            split_step,
                            stats
                        )
                    )
                    
                    
                    
                    # cur_dim = cur_depth%len(self.variables)
                    
                            
                    # splited_ranges = list(np.linspace(cur_bounds[cur_dim][0], cur_bounds[cur_dim][1], split_step+1))
                    
                    # for j in range(len(splited_ranges)-1):
                    
                    #     new_bounds = copy.deepcopy(cur_bounds)
                    #     new_bounds[cur_dim]=[splited_ranges[j], splited_ranges[j+1]]
                    #     self.to_check_queue.put((cur_depth+1, new_bounds, cur_volume/split_step, bound_vars, is_inside_list))
                        
                    # create two smaller hyper-rects
                    i = cur_depth%len(self.variables)
                    i_bounds = cur_bounds[i]
                    
                    s_bound_middle = (i_bounds[0]+i_bounds[1])/2
                    
                    left_bounds = copy.deepcopy(cur_bounds)
                    left_bounds[i]=[i_bounds[0], s_bound_middle]
                    self.to_check_queue.put((cur_depth+1, left_bounds, cur_volume/2, bound_vars, is_inside_list))
                    
                    right_bounds = copy.deepcopy(cur_bounds)
                    right_bounds[i]=[s_bound_middle, i_bounds[1]]
                    self.to_check_queue.put((cur_depth+1, right_bounds, cur_volume/2, bound_vars, is_inside_list))
            
            except Exception as e:
                logging.exception(e)
        
        logging.debug("I'm done!")

def calculate_approximate_volume(
        degree_list,
        max_workers,
        inputs, 
        bounds, 
        variables,
        threshold,
    ):

    global_start_time = time.time()


    # We apply Handelman below
    # The input form of Handelman is f_i>=0 => g_j >=0

    logging.info(f"Starting integral calculation with inputs: {inputs}, bounds: {bounds}, and Handelman degrees: {degree_list}")
    
    start_time = time.time()    
    equations = []
    equations_g = []
    # for each clause in RHS, we must apply handelmans
    for i, g_i in enumerate(inputs):

        # RHS

        # Generate symbolic f_is with symbolic variable for upper bound and lower bound
        f_list, bound_vars = generate_f_list(variables)

        # we apply handelman here to generate eqations based on l_is.
        # l_0 + l_1(f_1) + l_2(f_2) + ... + l_n(f_n) = g
        # equations.append(
            # Generate inside_equations, outside_equations, temp_vars
            # generate_handelman_equations(
            # degree=degree_list[i],
            # f_list=f_list,
            # g = g_i,
            # variables=variables
            # )
        # )
        
        equations_g.append(generate_handelman_equations_2(
            degree=degree_list[i],
            f_list=f_list,
            g = g_i,
            variables=variables
            ))
    
    logging.debug(f"Handelman equations generation time: {time.time()-start_time:.2f}s")

    
    # We run n instance of hardhats which listion on a queue and execute blocks
    # Create two queues
    to_check_queue = mp.Queue()
    checked_queue = mp.Queue()

    checker_list = []
    for _ in range(max_workers):
        checker = Checker(
            equations=None,
            equations_g=equations_g,
            variables=variables,
            to_check_queue=to_check_queue,
            checked_queue=checked_queue
        )
        checker.daemon=True
        checker.start()
        checker_list.append(checker)
    
    # We put the first hyper-rectangle
    start_volume = sym.prod([abs(b[1]-b[0]) for b in bounds])
     
    to_check_queue.put(
        (
            0, # start depth
            bounds, # starting bounds
            start_volume, #start volume
            bound_vars, # bound vars that need to be replaced,
            [False]*len(inputs) #Memoization
            # If a hyperrect is implies one literal, it will implies for all the smaller hyper rects which we create in future based on the current one
            
        )
    )
    
    error = start_volume # Start volume
    volume = 0
    
    total_hrect_checked = 0
    total_solver_time = 0
    total_subs_time = 0
    
    current_split_step = 2 
    
    while error > threshold and time.time()-global_start_time < 3600:
        res = checked_queue.get()

        total_hrect_checked += 1
        total_solver_time += res[-1]["solver_time"]
        total_subs_time += res[-1]["subs_time"]
        
        if res[0]==0:      
            error -= res[1]
            volume += res[1]
        elif res[0]==1:
            error -= res[1]
        elif res[0] ==2:
            current_split_step = max(current_split_step, res[-2])
        
        # To prevent excessive logging, we only log every 250 hrects
        if total_hrect_checked%50 == 0:
            logging.info(f"#HyperR Checked: {total_hrect_checked}, Error: {error:.6f}, Volume: ({volume:.6f},{volume+error:.6f}), Time: {time.time()-start_time:.2f}s")
                     
            logging.debug(f"Avg subs time: {(total_subs_time)/(total_hrect_checked*2):.6f}s, Avg solver time: {total_solver_time/(total_hrect_checked*2):.6f}s, Split step: {current_split_step}")


    # Final logging
    logging.info(f"#HyperR Checked: {total_hrect_checked}, Error: {error:.6f}, Volume: ({volume:.6f},{volume+error:.6f}), Time: {time.time()-start_time:.2f}s")
    
    if total_hrect_checked > 0:
        logging.debug(f"Avg subs time: {(total_subs_time)/(total_hrect_checked):.6f}s, Avg solver time: {total_solver_time/(total_hrect_checked):.6f}s")
    
    for checker in checker_list:
        checker.terminate()
        # checker.join()

    
    return volume, volume+error, {
        "hrect_checked_num": total_hrect_checked,
        "total_solver_time": total_solver_time,
        "total_subs_time": total_subs_time,
        "error": error        
    }

def find_upper_bound(    
    g, 
    variables,
    bounds
    ):
    
    
    f_list, bound_vars = generate_f_list(variables)
    
    
     # Optimization porblem
    subs_dict = {}
    for i in range(len(bounds)):
        subs_dict[bound_vars[i][0]] = bounds[i][0] 
        subs_dict[bound_vars[i][1]] = bounds[i][1]

    # Subsitute
    # we apply handelman here to generate eqations based on l_is.
    # l_0 + l_1(f_1) + l_2(f_2) + ... + l_n(f_n) = g
    
    # Since we need to find upper and lower bounds, we introduce two new variables
    U = sym.Symbol(f"U_{str(uuid.uuid4()).split('-')[0]}")


    # g_c = copy.deepcopy(g)
    # U_c = copy.deepcopy(U)
    # g <= U => U - g>=0
    # TODO: add proof rules
    # if sym.denom(g)!=1:
    #     n, d = sym.fraction(g)
    #     gU = U*d - n
    # else:
    #     gU = U - g
    # g <= U => U - g>=0
    
    # TODO: Fix this fucking shit, FUCK SYMPY    
    if isinstance(g, sym.core.power.Pow):
        root = int(1/g.args[-1])
        if root > 1:
            U = sym.expand(sym.Pow(U,root, evaluate=False))
            g = g.args[0]
    
    y = U*1
    if sym.denom(g)!=1:
        n, d = sym.fraction(g)
        y = sym.Mul(y,d, evaluate=False)
        g = n
        
    gU = sym.simplify(sym.expand(y-g))
    degree = sym.total_degree(gU)
            
    upper_equations, _, temp_vars = generate_handelman_equations(
        degree=degree,
        f_list=f_list,
        g = gU,
        variables=variables
    )
    equations = [
        sym.Poly(a.subs(subs_dict)) for a in upper_equations
        ]
    
    coeff_matrix = []
    for a in equations:
        coeff = []
        for v in [1]+temp_vars+[U]:
            try:
                coeff.append(float(a.coeff_monomial(v)))
            except:
                coeff.append(0)
        coeff_matrix.append(coeff)

    M = np.array(coeff_matrix)
    RHS = (M[:,:1]*-1).flatten()
    M = M[:,1:]
    
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            
            lp_l = m.addMVar(shape=len(temp_vars)+1, name="l", ub=[float('inf')]*len(temp_vars)+[float('inf')], lb=[0]*len(temp_vars)+[0])

            m.addConstr( M @ lp_l == RHS, name="c")

            m.setObjective(lp_l[-1], gp.GRB.MINIMIZE)            
            m.optimize()
            
            if m.status == gp.GRB.OPTIMAL:
                return True, float(lp_l[-1].x), m.runtime
            else:
                return False, None, m.runtime
          
def calculate_approximate_wmi(
        max_workers,
        w, 
        bounds,
        S, 
        variables,
        epsilon,
    ):

    start_time = time.time()

    # We apply Handelman below
    # The input form of Handelman is f_i>=0 => g >=0

    # Normalize the weight function
    if type(w) == str:
        w = sym.parse_expr(w, evaluate=False)
    
    variables = [sym.Symbol(v) for v in variables]
    
    ###### for psi+ ######
    has_upper_bound, upper_bound, runtime = find_upper_bound(
        g=w,
        bounds=bounds,
        variables=variables
    )
    
    if not has_upper_bound:
        raise Exception("Upper bound not found.")
    
    logging.info(f"Upper bound for weight function found: [0, {upper_bound}]")
    
    # We introduce a new variable
    y_symbol = sym.Symbol(f"y_{str(uuid.uuid4()).split('-')[0]}")


    y = y_symbol*1
    
    # TODO: Fix this fucking shit, FUCK SYMPY    
    if isinstance(w, sym.core.power.Pow):
        root = int(1/w.args[-1])
        if root > 1:
            y = sym.expand(sym.Pow(y,root, evaluate=False))
            w = w.args[0]
    
    if sym.denom(w)!=1:
        n, d = sym.fraction(w)
        
        new_integrand = n - sym.Mul(y,d, evaluate=False)
    else:
        new_integrand = w - y
        
    
    # # TODO: add proof rules
    # if sym.denom(w)!=1:
    #     n, d = sym.fraction(w)
    #     new_integrand = n - d*y
    # else:
    #     new_integrand = w - y
        
    new_bounds = bounds+[[0, upper_bound]]
    new_vars = variables+[y_symbol]
    inputs = [new_integrand]
    
    # we convert the inputs to the form g_i>0 or g_i >=0
    if (S != True) & (S != 1):
        for exp in S:
            if isinstance(exp, sym.core.relational.Lt) or isinstance(exp, sym.core.relational.Le):
                inputs.append(exp.rhs - exp.lhs)
            elif isinstance(exp, sym.core.relational.Gt) or isinstance(exp, sym.core.relational.Ge):
                inputs.append(exp.lhs-exp.rhs)

    
    
    inputs = [sym.expand(i) for i in inputs]    
    
    if upper_bound != 0:
        lower_psi_plus, upper_psi_plus, psi_plus_stats = calculate_approximate_volume(
            degree_list=[sym.total_degree(i) for i in inputs],
            max_workers=max_workers,
            inputs=inputs,
            bounds=new_bounds,
            variables=new_vars,
            threshold=epsilon
        )
    else:
        lower_psi_plus, upper_psi_plus = 0
        psi_plus_stats = {
        "hrect_checked_num": 0,
        "total_solver_time": 0,
        "total_subs_time": 0        
        }
    
    logging.info(f"Result: ({lower_psi_plus},{upper_psi_plus})")
    
    return lower_psi_plus, upper_psi_plus, {
        "hrect_checked_num": psi_plus_stats["hrect_checked_num"],
        "total_solver_time": psi_plus_stats["total_solver_time"],
        "total_subs_time": psi_plus_stats["total_subs_time"]        
    }


    