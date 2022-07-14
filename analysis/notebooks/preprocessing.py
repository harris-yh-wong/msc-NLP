import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def process_source_files(input_dir, output_dir):
    
    output_dir.mkdir(exist_ok=True)

    # input-output pairs
    filenames = [(f+'.txt', f+'_processed.txt') for f in ['test', 'train', 'dev']]
    io_pairs = [(input_dir/filename_in, output_dir/filename_out) for filename_in, filename_out in filenames]

    for in_file, out_file in io_pairs:
        pmid=''
        with open(in_file, 'r') as pmrctFile:
            with open(out_file, 'w') as outFile:
                n_line_read = 0
                n_line_written = 0
                for line in pmrctFile:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    if line.startswith('###'):
                        pmid=line.lstrip('###')
                    else:
                        outFile.writelines((pmid, '\t', line, '\n'))
                        n_line_written += 1
                    n_line_read += 1
                print(f'Read {n_line_read} lines from {in_file}')
                print(f'Wrote {n_line_written} lines to {out_file}')


def import_processed_files(dir):
    colnames = ['pmid', 'label', 'txt']
    train = pd.read_csv(dir/'train_processed.txt', sep='\t', header=None, names=colnames)
    dev   = pd.read_csv(dir/'dev_processed.txt', sep='\t', header=None, names=colnames)
    test  = pd.read_csv(dir/'test_processed.txt', sep='\t', header=None, names=colnames)
    return train, dev, test


def format_df(df):
    ### label dataframe
    df['sent_index'] = df.groupby(['pmid']).cumcount()
    ### index dataframe
    df.index = df['pmid'].astype(str) + '_' + df['sent_index'].astype(str)

    return df


def engineer_features(df):
    out = df.copy()
    out['label_prev'] = get_prev_label(out)
    return out


def get_prev_label(df):
    ### add a label for the previous instance
    prev_label = df.groupby('pmid')['label'].shift(1)
    prev_label = prev_label.fillna('FIRST', inplace=False)
    return prev_label
    

def get_spacy_lemmas(row):
    return [token.lemma_ for token in row.doc]


def get_spacy_pos(row):
    return [token.pos_ for token in row.doc]


def filter_spacy_stopwords(row):
    return [token for token in row.doc if not token.is_stop]


def _get_tfidf_model(
    docs: pd.Series, 
    **kwargs
):  
    ### original defaults
    # min_df = 5, 
    # max_df=100000, 
    # lowercase = True
    # stopwords = nltk.corpus.stopwords.words('english')

    tfidf_transformer = TfidfVectorizer(kwargs)
    tfidf_doc = tfidf_transformer.fit_transform(docs)
    feature_names = tfidf_transformer.get_feature_names() # list, # scipy.sparse.csr.csr_matrix
    return feature_names, tfidf_doc


def apply_spacy(docs: pd.Series, parser):
    d = dict()
    d['spacynlp'] = docs.apply(parser)
    # filtered = spacynlp.apply(preprocessing.filter_spacy_stopwords)
    d['lemma'] = d['spacynlp'].apply(get_spacy_lemmas)
    d['pos']   = d['spacynlp'].apply(get_spacy_pos)
    output_df = pd.DataFrame(d, index=docs.index)
    return output_df


def get_registries_regexes(f=None):
    """Extract registries and regexes information from the `ctregistries` github repo
    """
    if f is None:
        ### import registries information (containing databank name, identifier regex, etc)
        f = "https://raw.githubusercontent.com/maia-sh/ctregistries/master/inst/extdata/registries.csv"
    registries_df = pd.read_csv(f).query('databank_type == "registry"')

    ### skip EudraCT for now
    # regex does not work for this particular registry
    # suspected reason: R vs Python regex difference
    registries_regexes = registries_df[['databank', 'trn_regex']].query('databank != "EudraCT"')
    return registries_regexes


def flag_trial_identifier(docs: pd.Series, registries_regexes=None):
    ### import a dataframe with two columns:
    ### (i) databank name
    ### (ii) regex for identifier from this registry
    if registries_regexes is None:
        registries_regexes = get_registries_regexes()

    ### generate flags for each databank 
    ### save into a dictionary
    ### then combine to dataframe
    flags = dict()
    for databank, regex in zip(registries_regexes['databank'], registries_regexes['trn_regex']):
        flag = docs.str.contains(regex)
        flags[databank] = flag
    flags_trial_identifier_per_databank = pd.DataFrame(flags)

    ### search for the 'clinicaltrials.gov' string
    flag_clinical_trials_dot_gov = docs.str.contains(r"clinicaltrials.gov")

    ### combine
    flag_output = flags_trial_identifier_per_databank.any(axis=1) | flag_clinical_trials_dot_gov

    ### formatting
    flag_output = flag_output.rename("flag_any_trial_identifier")
    return flag_output


def flag_trial_registration(docs: pd.Series):    
    #### search for (i) date and (ii) contains the 'registry' keyword or synonyms
    date_regex = r"(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})"
    # search for dates in the format of '01 Jan 1970'
    flag_date = docs.str.contains(date_regex)
    flag_registration_keyword = docs.str.contains(r"registration|registry|registered")
    #! does not contain the word register
    
    ### intersection
    flag_output = flag_date & flag_registration_keyword
    
    ### format
    flag_output = flag_output.rename("flag_registration")
    return flag_output