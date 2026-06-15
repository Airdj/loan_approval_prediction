import yaml


def save_config_yaml(config, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False
        )


if __name__ == '__main__':
    from src.data.load_data import load_data
    import json
    import numpy as np
    import pandas as pd
    df_train, df_test, df_eval = load_data()

    my_rec = df_eval.iloc[0]
    my_rec = my_rec.replace({np.nan: None})
    my_rec = my_rec.where(pd.notnull(my_rec), None)
    json_st = my_rec.to_json(orient='records')
    record = (df_test.iloc[0])
    record = record.replace({np.nan: None})
    record = record.to_dict()

    json_str = json.dumps(record)
    print(json_str)