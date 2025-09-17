import os
import pandas as pd

COLUMN_NAMES = [
    'Customer Subtype',                # MOSTYPE
    'Number of houses',                # MAANTHUI
    'Avg size household',              # MGEMOMV
    'Avg age',                         # MGEMLEEF
    'Customer main type',              # MOSHOOFD
    'Roman catholic',                  # MGODRK
    'Protestant',                      # MGODPR
    'Other religion',                  # MGODOV
    'No religion',                     # MGODGE
    'Married',                         # MRELGE
    'Living together',                 # MRELSA
    'Other relation',                  # MRELOV
    'Singles',                         # MFALLEEN
    'Household without children',      # MFGEKIND
    'Household with children',         # MFWEKIND
    'High level education',            # MOPLHOOG
    'Medium level education',          # MOPLMIDD
    'Lower level education',           # MOPLLAAG
    'High status',                     # MBERHOOG
    'Entrepreneur',                    # MBERZELF
    'Farmer',                          # MBERBOER
    'Middle management',               # MBERMIDD
    'Skilled labourers',               # MBERARBG
    'Unskilled labourers',             # MBERARBO
    'Social class A',                  # MSKA
    'Social class B1',                 # MSKB1
    'Social class B2',                 # MSKB2
    'Social class C',                  # MSKC
    'Social class D',                  # MSKD
    'Rented house',                    # MHHUUR
    'Home owners',                     # MHKOOP
    '1 car',                           # MAUT1
    '2 cars',                          # MAUT2
    'No car',                          # MAUT0
    'National Health Service',         # MZFONDS
    'Private health insurance',        # MZPART
    'Income < 30.000',                 # MINKM30
    'Income 30-45.000',                # MINK3045
    'Income 45-75.000',                # MINK4575
    'Income 75-122.000',               # MINK7512
    'Income >123.000',                 # MINK123M
    'Average income',                  # MINKGEM
    'Purchasing power class',          # MKOOPKLA
    'Contribution private third party insurance', # PWAPART
    'Contribution third party insurance (firms)', # PWABEDR
    'Contribution third party insurane (agriculture)', # PWALAND
    'Contribution car policies',       # PPERSAUT
    'Contribution delivery van policies', # PBESAUT
    'Contribution motorcycle/scooter policies', # PMOTSCO
    'Contribution lorry policies',     # PVRAAUT
    'Contribution trailer policies',   # PAANHANG
    'Contribution tractor policies',   # PTRACTOR
    'Contribution agricultural machines policies', # PWERKT
    'Contribution moped policies',     # PBROM
    'Contribution life insurances',    # PLEVEN
    'Contribution private accident insurance policies', # PPERSONG
    'Contribution family accidents insurance policies', # PGEZONG
    'Contribution disability insurance policies', # PWAOREG
    'Contribution fire policies',      # PBRAND
    'Contribution surfboard policies', # PZEILPL
    'Contribution boat policies',      # PPLEZIER
    'Contribution bicycle policies',   # PFIETS
    'Contribution property insurance policies', # PINBOED
    'Contribution social security insurance policies', # PBYSTAND
    '# of private third party insurance', # AWAPART
    '# of third party insurance (firms)', # AWABEDR
    '# of third party insurane (agriculture)', # AWALAND
    '# of car policies',          # APERSAUT
    '# of delivery van policies', # ABESAUT
    '# of motorcycle/scooter policies', # AMOTSCO
    '# of lorry policies',        # AVRAAUT
    '# of trailer policies',      # AAANHANG
    '# of tractor policies',      # ATRACTOR
    '# of agricultural machines policies', # AWERKT
    '# of moped policies',        # ABROM
    '# of life insurances',       # ALEVEN
    '# of private accident insurance policies', # APERSONG
    '# of family accidents insurance policies', # AGEZONG
    '# of disability insurance policies', # AWAOREG
    '# of fire policies',         # ABRAND
    '# of surfboard policies',    # AZEILPL
    '# of boat policies',         # APLEZIER
    '# of bicycle policies',      # AFIETS
    '# of property insurance policies', # AINBOED
    '# of social security insurance policies', # ABYSTAND
    'CARAVAN'
]
assert len(COLUMN_NAMES) == 86

def load_caravan(data_dir="../data"):
    """
    Loads Caravan dataset and returns train, test DataFrames, X, y.
    data_dir: folder containing caravan.train and caravan.test
    """
    train_path = os.path.join(data_dir, "caravan.train")
    test_path  = os.path.join(data_dir, "caravan.test")

    train_raw = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python")
    test_raw  = pd.read_csv(test_path,  sep=r"\s+", header=None, engine="python")

    train = train_raw.copy(); test = test_raw.copy()
    train.columns = COLUMN_NAMES
    test.columns  = COLUMN_NAMES[:-1]  # test has no CARAVAN column

    TARGET = "CARAVAN"
    X = train.drop(columns=[TARGET])
    y = train[TARGET].astype(int)

    return train, test, X, y, TARGET

