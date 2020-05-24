import configuration as config
from sklearn.model_selection import train_test_split

class DataGenerator:
  LABEL_COLUMN = config.LABEL_COLUMN
  def __init__(self, data, split):
    labels = data.columns[:-1]

    X = data[labels]
    y = data[config.LABEL_COLUMN]

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split, random_state=config.RANDOM_SEED)
