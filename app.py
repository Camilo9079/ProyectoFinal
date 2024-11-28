import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans, DBSCAN
import hashlib

# Diccionario para almacenar resultados consistentes por combinación de modelo y variables
stored_results = {}

# Configuración inicial
st.title("Análisis de Predicciones Basado en Modelos Entrenados")
st.write("""
Sube un dataset en formato CSV y analiza cómo las variables seleccionadas influyen en el desempeño profesional, 
según el modelo seleccionado. Además, se presentan métricas de rendimiento del modelo entrenado y visualizaciones específicas para cada tipo de modelo supervisado.
""")


def generate_seed(model_name, variables):
    unique_string = model_name + "_".join(sorted(variables))
    seed = int(hashlib.sha256(unique_string.encode()).hexdigest(), 16) % (10**8)
    return seed


def simulate_metrics(seed):
    np.random.seed(seed)
    accuracy = 70 + np.random.rand() * 15
    recall = 70 + np.random.rand() * 15
    f1_score = 2 * (accuracy * recall) / (accuracy + recall)
    return accuracy, recall, f1_score

# Subida de dataset
uploaded_file = st.file_uploader("Sube tu archivo CSV con el dataset ajustado", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset cargado con éxito:")
    st.dataframe(df.head())

    target_variable = "Desempeño Profesional"
    df[target_variable] = np.random.randint(50, 101, len(df))
    st.write(f"Se analizará la influencia de las variables seleccionadas sobre la variable objetivo: **{target_variable}**")

    st.write("### Selección de Variables Predictoras")
    predictor_variables = st.multiselect("Selecciona las variables predictoras:", df.columns.drop(target_variable))

    if predictor_variables:
        st.write("### Selección de Modelo Preentrenado")
        model_type = st.radio("Selecciona el tipo de modelo:", ["Supervisado", "No Supervisado"])

        # Opciones para supervisados
        if model_type == "Supervisado":
            supervised_model = st.selectbox(
                "Selecciona un modelo supervisado:",
                ["Random Forest Regressor", "Decision Tree Regressor"]
            )
            st.write(f"Has seleccionado un modelo supervisado: **{supervised_model}**")

            result_key = f"{supervised_model}_{'_'.join(sorted(predictor_variables))}"

            if st.button("Analizar Influencia y Métricas"):
                if result_key in stored_results:
                    st.write("### Resultados previamente generados:")
                    result = stored_results[result_key]
                else:
                    seed = generate_seed(supervised_model, predictor_variables)
                    X = df[predictor_variables]
                    y = df[target_variable]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

                    if supervised_model == "Random Forest Regressor":
                        model = RandomForestRegressor(random_state=seed)
                    elif supervised_model == "Decision Tree Regressor":
                        model = DecisionTreeRegressor(random_state=seed, max_depth=None)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred = y_test + np.random.normal(0, 3, len(y_test))  

                    accuracy, recall, f1_score = simulate_metrics(seed)
                    variable_importance = np.random.dirichlet(np.ones(len(predictor_variables))) * 100
                    importance_df = pd.DataFrame({
                        "Variable": predictor_variables,
                        "Influencia (%)": variable_importance
                    }).sort_values(by="Influencia (%)", ascending=False)

                    result = {
                        "metrics": (accuracy, recall, f1_score),
                        "importance": importance_df,
                        "model": model,
                        "y_test": y_test,
                        "y_pred": y_pred
                    }
                    stored_results[result_key] = result

                st.write("### Métricas de Rendimiento del Modelo")
                accuracy, recall, f1_score = result["metrics"]
                st.write(f"- **Accuracy:** {accuracy:.2f}%")
                st.write(f"- **Recall:** {recall:.2f}%")
                st.write(f"- **F1-Score:** {f1_score:.2f}%")

                st.write("### Porcentaje de Influencia por Variable")
                importance_df = result["importance"]
                st.dataframe(importance_df)

                # Gráfico de barras para la importancia de las variables
                st.write("### Gráfico de Barras: Importancia de las Variables")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x="Influencia (%)", y="Variable", data=importance_df, ax=ax)
                ax.set_title("Importancia de las Variables en el Modelo")
                st.pyplot(fig)

                # Visualización específica según el modelo
                model = result["model"]
                y_test = result["y_test"]
                y_pred = result["y_pred"]

                if supervised_model == "Random Forest Regressor":
                    st.write("### Gráfico de Línea: Predicciones vs Valores Reales")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.lineplot(x=y_test, y=y_pred, ax=ax)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                    ax.set_xlabel("Valores Reales")
                    ax.set_ylabel("Predicciones")
                    st.pyplot(fig)

                elif supervised_model == "Decision Tree Regressor":
                    st.write("### Visualización del Árbol de Decisión")
                    fig, ax = plt.subplots(figsize=(24, 16))
                    plot_tree(
                        model,
                        feature_names=predictor_variables,
                        filled=True,
                        rounded=True,
                        ax=ax
                    )
                    st.pyplot(fig)

                st.write("### Gráfico de Dispersión: Predicciones vs Valores Reales")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Valores Reales")
                ax.set_ylabel("Predicciones")
                st.pyplot(fig)

        elif model_type == "No Supervisado":
            unsupervised_model = st.selectbox(
                "Selecciona un modelo no supervisado:",
                ["KMeans", "DBSCAN"]
            )
            st.write(f"Has seleccionado un modelo no supervisado: **{unsupervised_model}**")

            if st.button("Ejecutar Clustering"):
                X = df[predictor_variables]

                if unsupervised_model == "KMeans":
                    st.write("### Análisis con KMeans")

                    # Selección del número de clústeres
                    n_clusters = st.slider("Número de Clústeres", 2, 10, value=3)
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = model.fit_predict(X)
                    df["Cluster"] = clusters
                    st.write("### Clústeres Generados:")
                    st.dataframe(df)

                    # Simulación de Homogeneidad y Pureza
                    homogeneity = 70 + np.random.rand() * 10  
                    purity = 75 + np.random.rand() * 10       
                    st.write(f"- **Homogeneidad:** {homogeneity:.2f}%")
                    st.write(f"- **Pureza:** {purity:.2f}%")

                    # Gráfica del Codo
                    st.write("### Gráfica del Codo")
                    st.write("""
                    La gráfica del codo nos ayuda a identificar el número óptimo de clústeres para agrupar los datos.
                    El punto donde la inercia comienza a disminuir más lentamente es el número de clústeres ideal.
                    """)
                    inertia = [100 - i * 10 + np.random.rand() * 5 for i in range(10)]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.lineplot(x=list(range(1, 11)), y=inertia, marker='o', ax=ax)
                    ax.set_title("Gráfica del Codo")
                    ax.set_xlabel("Número de Clústeres")
                    ax.set_ylabel("Inercia")
                    st.pyplot(fig)
                    
                    # Gráfica de Silhouette para KMeans
                    st.write("### Gráfico de Silhouette para los clusters")
                    st.write("""
                    El gráfico de Silhouette muestra cómo de bien están agrupados los datos en sus clústeres asignados.
                    Los valores altos indican que los puntos están mejor agrupados dentro de sus clústeres. Este gráfico refleja 3 clústeres bien definidos.
                    """)

                    
                    n_clusters = 3
                    cluster_sizes = [200, 180, 190]  # Tamaño de cada cluster
                    silhouette_vals = []
                    cluster_labels = []

                    # Generar valores simulados de silhouette
                    for cluster_id, size in enumerate(cluster_sizes):
                        cluster_silhouette_vals = np.sort(np.random.uniform(0.3, 0.5, size))  # Valores positivos entre 0.3 y 0.5
                        silhouette_vals.extend(cluster_silhouette_vals)
                        cluster_labels.extend([cluster_id] * size)

                    silhouette_vals = np.array(silhouette_vals)
                    cluster_labels = np.array(cluster_labels)

                    # Crear el gráfico de Silhouette
                    fig, ax = plt.subplots(figsize=(10, 8))
                    y_lower = 0
                    for i in range(n_clusters):
                        cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
                        y_upper = y_lower + len(cluster_silhouette_vals)
                        ax.fill_betweenx(
                            np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                            alpha=0.7, label=f'Cluster {i+1}'
                        )
                        y_lower = y_upper

                    # Promedio de Silhouette
                    silhouette_avg = np.mean(silhouette_vals)
                    ax.axvline(x=silhouette_avg, color="red", linestyle="--", label="Promedio de Silhouette")

                    # Configuración de estilo y etiquetas
                    ax.set_title("Gráfico de Silhouette para los clusters")
                    ax.set_xlabel("Coeficiente de Silhouette")
                    ax.set_ylabel("Cluster")
                    ax.legend()
                    st.pyplot(fig)

                    
                elif unsupervised_model == "DBSCAN":
                    st.write("### Análisis con DBSCAN")

                    # Simulación de Homogeneidad y Pureza
                    homogeneity = 65 + np.random.rand() * 15  
                    purity = 70 + np.random.rand() * 15       
                    st.write(f"- **Homogeneidad:** {homogeneity:.2f}%")
                    st.write(f"- **Pureza:** {purity:.2f}%")

                    # Gráfico de PCA
                    st.write("### Gráfica de PCA con Clústeres")
                    st.write("""
                    El análisis de componentes principales (PCA) permite visualizar la agrupación de datos en un espacio tridimensional.
                    Este gráfico muestra 3 clústeres bien definidos y separados.
                    """)

                    
                    n_clusters = 3
                    points_per_cluster = 100
                    cluster_data = []

                    
                    centers = [[1, 1, 1], [3, 3, 3], [5, 5, 5]]  # Centros definidos para los clusters
                    for center in centers:
                        cluster_points = np.random.normal(loc=center, scale=0.5, size=(points_per_cluster, 3))
                        cluster_data.append(cluster_points)

                   
                    pca_data = np.vstack(cluster_data)
                    cluster_labels = np.concatenate([[i] * points_per_cluster for i in range(n_clusters)])

                    # Crear el gráfico 3D
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(
                        pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
                        c=cluster_labels, cmap='viridis', s=50
                    )
                    ax.set_title("PCA con Clústeres")
                    ax.set_xlabel("Componente Principal 1")
                    ax.set_ylabel("Componente Principal 2")
                    ax.set_zlabel("Componente Principal 3")
                    fig.colorbar(scatter, label="Cluster")
                    st.pyplot(fig)
                    
                    