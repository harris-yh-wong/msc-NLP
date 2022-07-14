def find_index(df, sent):
    flag = df['txt'].str.contains(sent)
    return df.loc[flag].index

    
def print_row(row):
    ncols = row.shape[0]
    lists = [row[i] for i in range(ncols)]

    ### check input data
    list_lengths = [len(l) for l in lists]
    assert len(set(list_lengths))==1, "Lists are of different lengths"
    
    ### print header
    for j in range(ncols):
        item = row.index[j]
        print(f'{item:<20}', end='')
    print('\n', end='')
    print('-'*20*ncols)
        
    ### print contents
    for obj in zip(*lists):
        for j in range(ncols):
            item = str(obj[j])
            print(f'{item:<20}', end='')
        print('\n', end='')


def compare_cleaned(df, max_row=20):
    nrow = df.shape[0]
    assert nrow<=max_row, "Too many rows."

    for i in range(nrow):
        row = df.iloc[i]
        print(row.name)
        print(row['txt'])
        print(row['clean'])
        print('\n', end='')


def preview(df, max_row=3, show_cols=('txt', ), header_cols=('index', 'label')):
    ncols = df.shape[0]
    if ncols >= max_row:
        print(f"max_row exceeded. Only showing {max_row} out of {ncols} rows.\n")
        df2 = df.head(max_row)
    else:
        df2 = df.copy()
        print('Showing {ncols} rows.\n')
        
    df2 = df2.reset_index(inplace=False)

    header_cols = [col for col in header_cols if col in df2.columns]

    for i in range(df2.shape[0]):
        row = df2.iloc[i]
        ### header
        for col in header_cols:
            print(' | ', end='')
            print(row[col], end='')
        print('\n', end='')

        ### contents
        for col in show_cols:
            print(row[col], end='\n')
        
        print('\n', end='')