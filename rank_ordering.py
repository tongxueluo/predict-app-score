__author__ = 'luoyuan'
'''this py file generate rank_order report. saved to './report'''''



import pandas as pd
import numpy as np

class rank_order:
    def __init__(self, output_path='./report'):
        self.output_path = output_path
    def rank_order(self, df, y_col, x_col, ascending=False, qcuts=10, version=0):
        df.sort_values(by=x_col, inplace=True, ascending=ascending)
        bins = np.unique( np.percentile(df[x_col], np.linspace(0, 100, qcuts+1), interpolation='lower') )
        bins = np.insert(bins,0,0)
        bins[-1] = 1
        df['grade'] = pd.tools.tile._bins_to_cuts(df[x_col], bins, include_lowest=False, labels=range(bins.shape[-1]-1,0,-1))

        cnt_tot = df.shape[0]
        bad_tot = df[y_col].sum()
        good_tot = cnt_tot - bad_tot

        CntCumGoods = 0
        CntCumBads =0
        attribute = []
        for i in range(1,qcuts+1):
            df_sub = df[df['grade'] == i].copy()
            min_proba = df_sub[x_col].min()
            max_proba = df_sub[x_col].max()

            Total = df_sub[y_col].count()
            NumOfBads = df_sub[y_col].sum()
            NumOfGoods = Total - NumOfBads

            CntCumBads += NumOfBads
            CntCumGoods += NumOfGoods

            if Total == 0:
                BadRate = 0
                GoodRate = 0
            else:
                BadRate = NumOfBads/ float(Total)
                GoodRate = NumOfGoods /float(Total)

            PctOfBads = NumOfBads / float(bad_tot)
            PctOfGoods = NumOfGoods / float(good_tot)

            CumPctOfBads = CntCumBads/ float(bad_tot)
            CumPctOfGoods = CntCumGoods / float(good_tot)

            HitRate = CumPctOfBads
            FalseAlarmRate = CumPctOfGoods

            KS = abs(CumPctOfBads - CumPctOfGoods)

            attribute.append( [round(min_proba,3), round(max_proba,3), NumOfGoods, NumOfBads, Total, CntCumBads, CumPctOfBads,
                               round(PctOfBads,3), round(GoodRate,3), CntCumGoods, CumPctOfGoods, round(PctOfGoods,3), round(HitRate,3),
                               round(FalseAlarmRate,3), round(KS,2), round(BadRate,3)])

        col = ['min_score', 'max_score', 'NumOfGoods', 'NumOfBads', 'Total',  'CntCumBads', 'CumPctOfBads', 'PctOfBads',\
                ' GoodRate', 'CntCumGoods', 'CumPctOfGoods', 'PctOfGoods', 'HitRate', 'FalseAlarmRate','KS','BadRate(Proba_of_Dfault)']

        pd.DataFrame(data=attribute, columns=col, index=range(1,qcuts+1)).to_csv(self.output_path+'/rank_order_%s.csv' % version,index_label='group')
