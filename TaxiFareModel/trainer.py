from sqlite3 import Time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipe = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder(time_column='pickup_datetime')),
                             ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder='drop')

        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

    def run(self):
        """set and train the pipeline"""

        self.set_pipeline()
        self.pipe.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        return compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    df = get_data()
    print('get')
    df = clean_data(df)
    print('clean')
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    model = Trainer(X_train, y_train)
    model.set_pipeline()
    model.run()
    print('model')
    score = model.evaluate(X_test, y_test)
    print(score)
