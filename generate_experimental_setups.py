import numpy as np
import pandas as pd
import math
import json

def log_base(x: float, base: float) -> float:
    if x <= 0:
        raise ValueError(f"log_base: x must be > 0, got {x}")
    if base <= 0 or base == 1.0:
        raise ValueError(f"log_base: base must be > 0 and != 1, got {base}")
    return math.log(x) / math.log(base)

def generate_RQ1_google_setups_v1():
    s_fixed = 0.9
    r_fixed = 0.9
    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "google-trace-data/cella_pdu6.csv"
    data_column_name = "measured_power_util"
    use_averages = True
    avg_val = pd.read_csv(data_source)[data_column_name].mean()
    std_val = pd.read_csv(data_source)[data_column_name].std()

    dicts_vec = []
    target_interval_vec = [
        (avg_val - std_val - 0.5*std_val, avg_val - std_val + 0.5*std_val),
        (avg_val + std_val - 0.5*std_val, avg_val + std_val + 0.5*std_val),
        (avg_val - 0.5*std_val, avg_val + 0.5*std_val),
        (avg_val - 0.1*std_val, avg_val + 0.1*std_val)
    ]

    for target_interval in target_interval_vec:
        (L,U) = target_interval
        eps = (U-L)*0.05
        avg_normalization_value = 1+(r_fixed/(1-r_fixed))+(s_fixed/(1-s_fixed)) if use_averages else 1

        

        T = np.ceil(log_base( (1-r_fixed)*eps/(2*avg_normalization_value) ,r_fixed))
        aux_params = {}
        aux_params["data_source"] = data_source
        aux_params["data_column_name"] = data_column_name
        aux_params["use_averages"] = use_averages
        aux_params["data_results"] = f"experimental-results/powertrace_L={L}_U={U}.csv"
        aux_params["infR"] = infR_fixed
        aux_params["supR"] = supR_fixed
        aux_params["r"] = r_fixed
        aux_params["s"] = s_fixed
        aux_params["eps"] = eps
        aux_params["T"] = T
        aux_params["L"] = L
        aux_params["U"] = U
        aux_params["save_csv"] = False

        dicts_vec.append(aux_params)
    
    with open("experimental-setups/RQ1-google.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)


def generate_RQ1_google_setups_eps_decrease():
    s_fixed = 0.9
    r_fixed = 0.9
    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "google-trace-data/cella_pdu6.csv"
    data_column_name = "measured_power_util"
    use_averages = True
    avg_val = pd.read_csv(data_source)[data_column_name].mean()
    std_val = pd.read_csv(data_source)[data_column_name].std()

    dicts_vec = []
    target_interval = avg_val - 0.5*std_val, avg_val + 0.5*std_val
    eps_factors = np.logspace(np.log10(0.5), np.log10(0.0005), 8)



    for eps in eps_factors:
        (L,U) = target_interval
        avg_normalization_value = 1+(r_fixed/(1-r_fixed))+(s_fixed/(1-s_fixed)) if use_averages else 1
        T = np.ceil(log_base( (1-r_fixed)*eps/(2*avg_normalization_value) ,r_fixed))
        aux_params = {}
        aux_params["data_source"] = data_source
        aux_params["data_column_name"] = data_column_name
        aux_params["data_results"] = f"experimental-results/powertrace_eps_{eps}.csv"
        aux_params["use_averages"] = use_averages
        aux_params["infR"] = infR_fixed
        aux_params["supR"] = supR_fixed
        aux_params["r"] = r_fixed
        aux_params["s"] = s_fixed
        aux_params["eps"] = eps
        aux_params["T"] = T
        aux_params["L"] = L
        aux_params["U"] = U
        aux_params["save_csv"] = False

        dicts_vec.append(aux_params)
    
    with open("experimental-setups/RQ1-google-decrease-eps.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)


def generate_RQ1_google_setups_targetinterval_decrease():
    s_fixed = 0.9
    r_fixed = 0.9
    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "google-trace-data/cella_pdu6.csv"
    data_column_name = "measured_power_util"
    use_averages = True
    avg_val = pd.read_csv(data_source)[data_column_name].mean()
    std_val = pd.read_csv(data_source)[data_column_name].std()

    dicts_vec = []
    width_factors = np.logspace(np.log10(10), np.log10(0.01), 8)
    eps = 0.0005


    for width_factor in width_factors:
        L = avg_val - width_factor*std_val
        U = avg_val + width_factor*std_val
        width = 2*width_factor*std_val

        avg_normalization_value = 1+(r_fixed/(1-r_fixed))+(s_fixed/(1-s_fixed)) if use_averages else 1
        T = np.ceil(log_base( (1-r_fixed)*eps/(2*avg_normalization_value) ,r_fixed))
        aux_params = {}
        aux_params["data_source"] = data_source
        aux_params["data_column_name"] = data_column_name
        aux_params["data_results"] = f"experimental-results/powertrace_interval_length_{width}.csv"
        aux_params["use_averages"] = use_averages
        aux_params["infR"] = infR_fixed
        aux_params["supR"] = supR_fixed
        aux_params["r"] = r_fixed
        aux_params["s"] = s_fixed
        aux_params["eps"] = eps
        aux_params["T"] = T
        aux_params["L"] = L
        aux_params["U"] = U
        aux_params["save_csv"] = False

        dicts_vec.append(aux_params)
    
    with open("experimental-setups/RQ1-google-decrease-target-interval.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)


def generate_adult_race_decrease_eps():
    s_fixed = 0.95
    r_fixed = 0.95
    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "ffb_traces/adult_race_hsic.csv"
    data_column_name = "input"
    use_averages = False

    dicts_vec = []
    target_interval = -0.1, 0.1

    eps_factors = np.logspace(np.log10(0.05), np.log10(0.0005), 8)

    for use_synchronous in [False, True]:
        for eps in eps_factors:
            (L,U) = target_interval
            T = 10*np.ceil(log_base( (1-r_fixed)*eps/(2) ,r_fixed))
            aux_params = {}
            aux_params["data_source"] = data_source
            aux_params["data_column_name"] = data_column_name
            sync_name = "sync" if use_synchronous else "async"
            aux_params["data_results"] = f"experimental-results/adult_race_hsic_eps_{eps}_{sync_name}.csv"
            aux_params["use_averages"] = use_averages
            aux_params["infR"] = infR_fixed
            aux_params["supR"] = supR_fixed
            aux_params["r"] = r_fixed
            aux_params["s"] = s_fixed
            aux_params["eps"] = eps
            aux_params["T"] = T
            aux_params["L"] = L
            aux_params["U"] = U
            aux_params["save_csv"] = False
            aux_params["synchronous"] = use_synchronous

            dicts_vec.append(aux_params)
    
    with open("experimental-setups/ffb-adult-race-decrease-eps.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)



def generate_adult_race_decrease_interval_length():
    s_fixed = 0.95
    r_fixed = 0.95
    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "ffb_traces/adult_race_hsic.csv"
    data_column_name = "input"
    use_averages = False

    dicts_vec = []
    width_factors = np.logspace(np.log10(0.25), np.log10(0.0025), 8)
    eps = 0.00005

    for use_synchronous in [False, True]:
        for width_factor in width_factors:
            L = -width_factor
            U = width_factor
            width = width_factor
            T = 5*np.ceil(log_base( (1-r_fixed)*eps/(2) ,r_fixed))
            aux_params = {}
            aux_params["data_source"] = data_source
            aux_params["data_column_name"] = data_column_name
            sync_name = "sync" if use_synchronous else "async"
            aux_params["data_results"] = f"experimental-results/adult_race_hsic_interval_length_{width}_{sync_name}.csv"
            aux_params["use_averages"] = use_averages
            aux_params["infR"] = infR_fixed
            aux_params["supR"] = supR_fixed
            aux_params["r"] = r_fixed
            aux_params["s"] = s_fixed
            aux_params["eps"] = eps
            aux_params["T"] = T
            aux_params["L"] = L
            aux_params["U"] = U
            aux_params["save_csv"] = False
            aux_params["synchronous"] = use_synchronous

            dicts_vec.append(aux_params)
    
    with open("experimental-setups/ffb-adult-race-decrease-interval-length.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)



def generate_stochastic_decrease_interval_length():
    s_fixed = 0.95
    r_fixed = 0.95
    supR_fixed = 1.0
    infR_fixed = 0.0

    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "stochastic_traces/sin1000.csv"
    data_column_name = "input"
    use_averages = True

    dicts_vec = []
    width_factors = np.logspace(np.log10(0.25), np.log10(0.0025), 8)
    eps = 0.001

    for j in range(1):

        for width_factor in width_factors:
            for uniform in [False,True]:
                L = 0.5 - width_factor
                U = 0.5 + width_factor
                width = width_factor
                T = 1000
                aux_params = {}
                data_source = f"stochastic_traces/sin1000_{j}.csv"
                aux_params["data_source"] = data_source
                aux_params["data_column_name"] = data_column_name
                aux_params["data_results"] = f"experimental-results/stochastic_interval_length_uniform_{uniform}_{width}_{j}.csv"
                aux_params["use_averages"] = use_averages
                aux_params["infR"] = infR_fixed
                aux_params["supR"] = supR_fixed
                aux_params["r"] = r_fixed
                aux_params["s"] = s_fixed
                aux_params["eps"] = eps
                aux_params["T"] = T
                aux_params["L"] = L
                aux_params["U"] = U
                aux_params["uniform"] = uniform
                aux_params["save_csv"] = False

                dicts_vec.append(aux_params)

    with open("experimental-setups/stochastic-decrease-interval-length.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)


def generate_stochastic_decrease_interval_length_rand():
    s_fixed = 0.95
    r_fixed = 0.95
    supR_fixed = 1.0
    infR_fixed = 0.0

    supR_fixed = 1.0
    infR_fixed = 0.0
    data_column_name = "input"
    use_averages = True

    dicts_vec = []
    width_factors = np.logspace(np.log10(0.25), np.log10(0.0025), 8)
    eps = 0.001

    for j in range(1):

        for width_factor in width_factors:
            for uniform in [False,True]:
                L = 0.5 - width_factor
                U = 0.5 + width_factor
                width = width_factor
                T = 1000
                aux_params = {}
                data_source = f"stochastic_traces/rand_{j}.csv"
                aux_params["data_source"] = data_source
                aux_params["data_column_name"] = data_column_name
                aux_params["data_results"] = f"experimental-results/stochastic_interval_length_rand_uniform_{uniform}_{width}_{j}.csv"
                aux_params["use_averages"] = use_averages
                aux_params["infR"] = infR_fixed
                aux_params["supR"] = supR_fixed
                aux_params["r"] = r_fixed
                aux_params["s"] = s_fixed
                aux_params["eps"] = eps
                aux_params["T"] = T
                aux_params["L"] = L
                aux_params["U"] = U
                aux_params["uniform"] = uniform
                aux_params["save_csv"] = False

                dicts_vec.append(aux_params)

    with open("experimental-setups/stochastic-decrease-interval-length_rand.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)



def generate_stochastic_decrease_interval_length30k():
    s_fixed = 0.9
    r_fixed = 0.9
    supR_fixed = 1.0
    infR_fixed = 0.0

    supR_fixed = 1.0
    infR_fixed = 0.0
    data_source = "stochastic_traces/sin1000.csv"
    data_column_name = "input"
    use_averages = True

    dicts_vec = []
    width_factors = np.logspace(np.log10(0.25), np.log10(0.0025), 8)
    eps = 0.0005

    for j in range(1):

        for width_factor in width_factors:
            for uniform in [False]:
                L = 0.5 - width_factor
                U = 0.5 + width_factor
                width = width_factor
                T = 1000
                aux_params = {}
                data_source = f"stochastic_traces/sin1000_30k_{j}.csv"
                aux_params["data_source"] = data_source
                aux_params["data_column_name"] = data_column_name
                aux_params["data_results"] = f"experimental-results/stochastic_interval_length30k_uniform_{uniform}_{width}_{j}.csv"
                aux_params["use_averages"] = use_averages
                aux_params["infR"] = infR_fixed
                aux_params["supR"] = supR_fixed
                aux_params["r"] = r_fixed
                aux_params["s"] = s_fixed
                aux_params["eps"] = eps
                aux_params["T"] = T
                aux_params["L"] = L
                aux_params["U"] = U
                aux_params["uniform"] = uniform
                aux_params["save_csv"] = False

                dicts_vec.append(aux_params)

    with open("experimental-setups/stochastic-decrease-interval-length30k.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)


def generate_stochastic_decrease_interval_length_rand30k():
    s_fixed = 0.9
    r_fixed = 0.9
    supR_fixed = 1.0
    infR_fixed = 0.0

    supR_fixed = 1.0
    infR_fixed = 0.0
    data_column_name = "input"
    use_averages = True

    dicts_vec = []
    width_factors = np.logspace(np.log10(0.25), np.log10(0.0025), 8)
    eps = 0.0005

    for j in range(1):

        for width_factor in width_factors:
            for uniform in [False]:
                L = 0.5 - width_factor
                U = 0.5 + width_factor
                width = width_factor
                T = 1000
                aux_params = {}
                data_source = f"stochastic_traces/rand_30k_{j}.csv"
                aux_params["data_source"] = data_source
                aux_params["data_column_name"] = data_column_name
                aux_params["data_results"] = f"experimental-results/stochastic_interval_length30k_rand_uniform_{uniform}_{width}_{j}.csv"
                aux_params["use_averages"] = use_averages
                aux_params["infR"] = infR_fixed
                aux_params["supR"] = supR_fixed
                aux_params["r"] = r_fixed
                aux_params["s"] = s_fixed
                aux_params["eps"] = eps
                aux_params["T"] = T
                aux_params["L"] = L
                aux_params["U"] = U
                aux_params["uniform"] = uniform
                aux_params["save_csv"] = False

                dicts_vec.append(aux_params)

    with open("experimental-setups/stochastic-decrease-interval-length30k_rand.json", "w") as fp:
        json.dump(dicts_vec, fp, indent=2)





def main():
    generate_RQ1_google_setups_v1()
    generate_RQ1_google_setups_eps_decrease()
    generate_RQ1_google_setups_targetinterval_decrease()
    generate_adult_race_decrease_eps()
    generate_adult_race_decrease_interval_length()
    generate_stochastic_decrease_interval_length()
    generate_stochastic_decrease_interval_length_rand()

    generate_stochastic_decrease_interval_length30k()
    generate_stochastic_decrease_interval_length_rand30k()
    

if __name__ == "__main__":
    main()