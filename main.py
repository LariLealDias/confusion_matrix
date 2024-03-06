import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data_df = pd.read_csv('./data/diabetes.csv')

x = data_df.drop('diabetes', axis=1)
y = data_df['diabetes']

x, x_test, y, y_test = train_test_split(x, y, test_size=0.5, stratify=y, random_state=5)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, stratify=y, random_state=5)

decision_tree = DecisionTreeClassifier(max_depth=2)

decision_tree.fit(x_train, y_train)
print(f"Acurácia de train: {decision_tree.score(x_train, y_train)}")
print(f"Acurácia de val: {decision_tree.score(x_val, y_val)}")

y_predict = decision_tree.predict(x_val)
generate_confusion_matrix = confusion_matrix(y_val, y_predict)

print(generate_confusion_matrix)

generate_confusion_matrix_graphic = ConfusionMatrixDisplay(confusion_matrix=generate_confusion_matrix, display_labels=['n-diabetico', 'diabetico'])
generate_confusion_matrix_graphic.plot()
plt.show()


