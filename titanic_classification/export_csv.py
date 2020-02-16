import os


def export_csv(df, outdir):
    if not os.path.exists(outdir):
        os.mkdir(f'{os.getcwd()}/{outdir}')
    df.to_csv(index=False, path_or_buf=f'{os.getcwd()}/{outdir}/submission.csv')
