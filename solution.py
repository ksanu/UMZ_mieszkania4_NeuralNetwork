import pandas
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder


def encoder(data):
    '''Map the categorical variables to numbers to work with scikit learn'''
    for col in data.columns:
        if data.dtypes[col] == "object":
            data[col].fillna('0', inplace=True)
            le = LabelEncoder()
            le.fit(data[col])
            data[col] = le.transform(data[col])
        else:
            data[col].fillna(0, inplace=True)

    return data


X_train = pandas.read_csv('train/train.tsv', error_bad_lines=False, sep='\t', header=0)
names = list(X_train)[1:]
X_train.drop('opis', axis=1, inplace=True)
y_train = X_train['cena']
X_train.drop('cena', axis=1, inplace=True)

X_train = encoder(X_train)


X_dev0 = pandas.read_csv('dev-0/in.tsv', sep='\t', header=None, names=names)
X_dev0.drop('opis', axis=1, inplace=True)
X_dev0 = encoder(X_dev0)
y_dev0 = pandas.read_csv('dev-0/expected.tsv', sep='\t', header=None)

X_testA = pandas.read_csv('test-A/in.tsv', sep='\t', header=None, names=names)
X_testA.drop('opis', axis=1, inplace=True)
X_testA = encoder(X_testA)

neuralNetwork = MLPRegressor(solver='lbfgs')
model = neuralNetwork.fit(X_train, y_train)

y_out_dev0 = model.predict(X_dev0)
y_out_testA = model.predict(X_testA)

with open('dev-0/out.tsv', 'w') as output_file:
    for out in y_out_dev0:
        print('%.0f' % out, file = output_file)

with open('test-A/out.tsv', 'w') as output_file:
    for out in y_out_testA:
        print('%.0f' % out, file = output_file)
