import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  # Depreciado, exibirá um alerta
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IPython.display import Image
import pydotplus
import os

# Comando para encontrar o caminho certo do Graphviz, a fim de evitar erros
# Command to find Graphviz's correct path, as to avoid errors
# Requer o Graphviz instalado!
# Requires Graphviz installed!
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


def create_graph(clf, feature_cols):
    # Declarando uma variável para o processo de criação de imagem
    # Declaring a variable for the image creation process
    dot_data = StringIO()

    # Utilizando a função para exportar os dados como um grafo do Graphviz
    # Using a function to export data as a Graphviz graph
    # class_names conterá 0 e 1 para indicar se a estrela é ou não uma pulsar
    # class_names will contain 0 and 1 to indicate if a star is a pulsar or not
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=feature_cols, class_names=['0', '1'])

    # Declarando a variável grafo em si, que será construido a partir do StringIO
    # Declaring the graph variable, which will be created from StringIO
    # O grafo será, então, convertido para um .png e a imagem será salva na pasta do projeto
    # The graph will, then, be converted into a .png and the image will be saved in the project folder
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('pulsar.png')
    Image(graph.create_png())


def train_decision_tree(X, y, feature_cols):
    # Dividindo o dataset para treino e teste
    # Dividing the dataset between training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% para treino e 30% para teste

    # Escolhendo o classifier, seu critério de otimização e treinando-o
    # Choosing the classifier, its optimization criterion and training it
    # O entropy é utilizado para mensurar a "pureza" dos dados de input
    # Entropy is used for measuring the "purity" of input data
    # O ganho de informações diminui o valor de entropy (irá reduzir o tamanho da árvore)
    # Information gain will decrease the entropy value (will reduce the overall tree size)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Verificando a precisão do modelo
    print("\nAccuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))

    create_graph(clf, feature_cols)


def main():
    pulsar = pd.read_csv("pulsar_stars.csv")

    # Verificando as primeiras linhas do dataset
    # Verifying the first rows from the dataset
    print(pulsar.head())

    # Nomes das colunas de features
    # Feature column names
    feature_cols = ["Mean of the integrated profile", "Standard deviation of the integrated profile", "Excess kurtosis of the integrated profile",
                    "Skewness of the integrated profile", "Mean of the DM-SNR curve", "Standard deviation of the DM-SNR curve",
                    "Excess kurtosis of the DM-SNR curve", "Skewness of the DM-SNR curve"]

    # Definindo as variáveis que serão utilizadas
    # Defining the variables that'll be used
    # Onde y é o valor que se deseja encontrar (uma estrela é ou não uma pulsar)
    # Where y is the target value (a star is or isn't a pulsar)
    X = pulsar[feature_cols]
    y = pulsar.target_class

    train_decision_tree(X, y, feature_cols)


# Definindo a função main() como a primeira função a ser chamada quando o programa for executado
# Defining the main() function as the first function to be called when the program is launched
if __name__ == "__main__":
    main()

