import pandas as pd


def main():
    m = pd.read_csv('outputs/metrics.csv')
    e = pd.read_csv('outputs/economics.csv')
    agg_dict = {
        'MAE':'mean','RMSE':'mean','MAPE':'mean','WAPE':'mean','sMAPE':'mean','MdAPE':'mean'
    }
    if 'MASE' in m.columns:
        agg_dict['MASE'] = 'mean'
    m_agg = m.groupby(['Tk','Model']).agg(**{k:(k,v) for k,v in agg_dict.items()}).reset_index()
    e_agg = e.groupby(['Tk','Model']).agg(CumRet=('CumRet','mean'), MaxDD=('MaxDD','mean')).reset_index()
    res = m_agg.merge(e_agg, on=['Tk','Model'], how='left')

    print('=== Aggregate metrics per ticker/model ===')
    print(res.sort_values(['Tk','MAE']).to_string(index=False))

    print('\n=== Best by MAE per ticker ===')
    print(res.loc[res.groupby('Tk')['MAE'].idxmin()].to_string(index=False))

    print('\n=== Best by WAPE per ticker ===')
    print(res.loc[res.groupby('Tk')['WAPE'].idxmin()].to_string(index=False))

    try:
        dm = pd.read_csv('outputs/dm_test_pairs.csv')
        print('\n=== Pairwise DM tests (first 20 rows) ===')
        print(dm.head(20).to_string(index=False))
    except Exception as exc:
        print('No pairwise DM file:', exc)


if __name__ == '__main__':
    main()
