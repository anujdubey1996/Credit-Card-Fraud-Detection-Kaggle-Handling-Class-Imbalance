import configuration as config
import pandas as pd

def percentage(part, whole):
  return 100 * float(part) / float(whole)

def get_data(filename):
    file = pd.read_csv(filename)
    print("Data loaded from input CSV")
    print("Shape of input file:" + str(file.shape))
    return file

def plot_class_balance(df, target, title):
    df[target].value_counts().plot(kind="bar", title= title);