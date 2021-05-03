import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Reinforcement_Model_Data.csv')
df['Recommendations'] = df['Recommendations'].astype('category')
df['Recommendations'] = df['Recommendations'].cat.codes


ax = sns.countplot(x='Recommendations', data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

print(df.corr())
sns.heatmap(df.corr())
plt.tight_layout()
plt.show()

X = df.drop(['Recommendations', 'Mood Score', 'Unnamed: 0'], axis=1).values
y = df['Recommendations']

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
print(df.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(66)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

for i in [10, 20, 50, 100, 400, 800]:
    # Fit model
    model.fit(X_train, y_train, epochs=i)
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    # make predictions, convert to data frame
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(X_test)
    solutions = pd.DataFrame(data = y_test)
    pre = []
    for j in range(len(predictions)):
        pre.append(np.argmax(predictions[j]))
    solutions['Predictions'] = pre

    print(solutions)

    # Create classification Report
    from sklearn.metrics import classification_report

    report = classification_report(y_test, solutions['Predictions'], output_dict=True)
    class_report = pd.DataFrame(report).transpose()
    print(class_report)
    input('Press ENTER to continue')

# solutions.to_csv('Reinforcement_Model_Data_Evaluation_50_epochs.csv')
# class_report.to_csv('Reinforcement_Model_Data_Classification_Report_50_epochs.csv')

# print(np.argmax(predictions[3]))
