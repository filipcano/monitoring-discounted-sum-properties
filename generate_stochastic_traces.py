import pandas as pd
import numpy as np


def generate_sin_traces():
    r = 0.9
    s = 0.9

    for j in range(1):
        times = np.arange(0,30000)
        ps = np.sin((np.pi/1000)*times)**2
        xs = 0*ps
        for i in range(len(times)):
            p = ps[i]
            xs[i] = 1 if np.random.rand() < p else 0
        df_dict = {
            "time" : times,
            "p" : ps,
            "input" : xs
        }
        df = pd.DataFrame(df_dict)
        p = df["p"]
        n = len(p)

        # past contributions
        past = sum(
            (s**i) * p.shift(i)
            for i in range(1, n)
        )

        # future contributions
        future = sum(
            (r**i) * p.shift(-i)
            for i in range(1, n)
        )

        df["discounted_p"] = p + past.fillna(0) + future.fillna(0)
        df.to_csv(f"stochastic_traces/sin1000_30k_{j}.csv")


def generate_rand_traces():
    r = 0.9
    s = 0.9


    for j in range(1):
        times = np.arange(0,30000)
        ps = np.sin((np.pi/1000)*times)**2
        xs = 0*ps
        ps = 0*ps
        for i in range(len(times)):
            ps[i] = np.random.rand()
            xs[i] = 1 if np.random.rand() < ps[i] else 0
        df_dict = {
            "time" : times,
            "p" : ps,
            "input" : xs
        }
        df = pd.DataFrame(df_dict)
        p = df["p"]
        n = len(p)

        # past contributions
        past = sum(
            (s**i) * p.shift(i)
            for i in range(1, n)
        )

        # future contributions
        future = sum(
            (r**i) * p.shift(-i)
            for i in range(1, n)
        )

        df["discounted_p"] = p + past.fillna(0) + future.fillna(0)
        df.to_csv(f"stochastic_traces/rand_30k_{j}.csv")


def main():
    generate_sin_traces()    
    generate_rand_traces()
    


if __name__ == "__main__":
    main()