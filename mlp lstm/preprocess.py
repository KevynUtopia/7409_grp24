
# preprocess raw csv data

import pandas as pd


def main():
    # df = pd.read_csv('./BTCUSD_1h.csv')
    raw_df = pd.read_csv('./ETHUSDT_1h.csv')
    # df = raw_df.sort_values('Unix')
    df = raw_df.iloc[::-1]
    # df = raw_df.reindex(index=raw_df.index[::-1])
    print(df.columns)
    print(df.head)
    pass


if __name__ == "__main__":   
    
    main()


