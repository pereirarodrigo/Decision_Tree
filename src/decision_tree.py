import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  # Depreciado, exibirá um alerta
from sklearn.model_selection import train_test_split
from sklearn import metrics
from IPython.display import Image
import pydotplus
import os

# Para encontrar o caminho certo do Graphviz, a fim de evitar erros
# Requer o Graphviz instalado!
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"


def create_graph(clf, feature_cols):
    # Declarando uma variável para criar a imagem
    dot_data = StringIO()

    # Utilizando a função para exportar os dados como um grafo do Graphviz
    # class_names conterá 0 e 1 para indicar se a estrela é ou não uma pulsar
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True,
                    feature_names=feature_cols, class_names=['0', '1'])

    # Declarando o grafo em si, que será construido a partir do StringIO
    # O grafo será, então, convertido para um arquivo png e a imagem será salva na pasta do projeto
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('pulsar.png')
    Image(graph.create_png())


def train_decision_tree(X, y, feature_cols):
    # Dividindo o dataset para treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)  # 70% para treino e 30% para teste

    # Escolhendo o classifier, seu critério de otimização e treinando-o
    # O entropy é utilizado para mensurar a "pureza" dos dados de input
    # O ganho de informações diminui o valor de entropy (irá reduzir o tamanho da árvore)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Verificando a precisão do modelo
    print("\nAccuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))

    create_graph(clf, feature_cols)


def main():
    pulsar = pd.read_csv("pulsar_stars.csv")

    # Verificando os primeiros registros do dataset
    print(pulsar.head())

    # Nomes das colunas de features
    feature_cols = ["Mean of the integrated profile", "Standard deviation of the integrated profile", "Excess kurtosis of the integrated profile",
                    "Skewness of the integrated profile", "Mean of the DM-SNR curve", "Standard deviation of the DM-SNR curve",
                    "Excess kurtosis of the DM-SNR curve", "Skewness of the DM-SNR curve"]

    # Definindo as variáveis que serão utilizadas
    # Onde y é o valor que se deseja encontrar (é ou não uma pulsar)
    X = pulsar[feature_cols]
    y = pulsar.target_class

    train_decision_tree(X, y, feature_cols)


if __name__ == "__main__":
    main()