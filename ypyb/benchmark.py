import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def save_to_firestore(collection_name, document_id, data):
    """
    Save data to a Firestore collection.
    """
    try:
        doc_ref = db.collection(collection_name).document(document_id)
        doc_ref.set(data)
        print(f"Document {document_id} successfully written to {collection_name}.")
    except Exception as e:
        print(f"Error writing to Firestore: {e}")


def read_from_firestore(collection_name, document_id):
    """
    Read data from a Firestore collection.
    """
    try:
        doc_ref = db.collection(collection_name).document(document_id)
        doc = doc_ref.get()
        if doc.exists:
            print(f"Document data: {doc.to_dict()}")
            return doc.to_dict()
        else:
            print(f"No such document {document_id} in collection {collection_name}.")
            return None
    except Exception as e:
        print(f"Error reading from Firestore: {e}")
        return None


def upload_to_storage(bucket_name, source_file_path, destination_blob_name):
    """
    Upload a file to Google Cloud Storage.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        print(f"File {source_file_path} uploaded to {destination_blob_name} in bucket {bucket_name}.")
    except Exception as e:
        print(f"Error uploading to Cloud Storage: {e}")


def download_from_storage(bucket_name, source_blob_name, destination_file_path):
    """
    Download a file from Google Cloud Storage.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_path)
        print(f"File {source_blob_name} downloaded to {destination_file_path} from bucket {bucket_name}.")
    except Exception as e:
        print(f"Error downloading from Cloud Storage: {e}")


def read_all_documents_from_collection(db, collection_name):
    """
    Read all documents from a specified Firestore collection.
    
    :param collection_name: The name of the Firestore collection to read from.
    :return: A list of dictionaries representing the documents in the collection.
    """
    try:
        collection_ref = db.collection(collection_name)
        documents = collection_ref.stream()
        
        all_docs = []
        for doc in documents:
            print(f"Document ID: {doc.id}, Data: {doc.to_dict()}")
            all_docs.append({"id": doc.id, **doc.to_dict()})
        
        return all_docs
    except Exception as e:
        print(f"Error reading documents from collection {collection_name}: {e}")
        return []

import os

def display_table_with_matplotlib(dataframe):
    fig, ax = plt.subplots(figsize=(10, len(dataframe) + 2))  # Adjust figure size based on data
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = plt.table(cellText=dataframe.values,
                      colLabels=dataframe.columns,
                      cellLoc='center',
                      loc='center',
                      colLoc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))

    # Add a title
    plt.title("Model Metrics Summary", fontsize=14, weight='bold', pad=20)
    print(f'{os.getcwd()}/metricas_deteccion.png')
    plt.savefig(f'{os.getcwd()}/metricas_deteccion.png', dpi=300)  # Ajusta la resolución de la imagen

    plt.show()


def display_table_with_styled_headers(dataframe):

    # Asumimos que 'dataframe' es el DataFrame que deseas visualizar en forma de tabla

    fig, ax = plt.subplots(figsize=(10, len(dataframe) + 2))  # Ajusta el tamaño de la figura
    ax.axis('tight')
    ax.axis('off')

    # Crear la tabla
    table = plt.table(cellText=dataframe.values,
                    colLabels=dataframe.columns,
                    cellLoc='center',
                    loc='center',
                    colLoc='center')

    # Estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(dataframe.columns))))

    # Estilo de la fila de encabezado (fondo gris)
    header_color = mcolors.CSS4_COLORS["lightgray"]
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Fila de encabezado
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold")  # Texto en negrita en el encabezado

    # Agregar un título
    # plt.title("Model Metrics Summary (Ordered by mAP@0.5)", fontsize=14, weight='bold', pad=20)

    # Guardar la imagen
    output_path = os.path.join(os.getcwd(), 'metricas_deteccion.png')
    print(output_path)
    plt.savefig(output_path, dpi=500, bbox_inches='tight', pad_inches=0.1)  # Ajustar el borde y el padding

    # Mostrar la gráfica
    plt.show()


def run():

    service_account_json_path = '/home/pqbas/labinm/PE501086701-2024-PROCIENCIA/blueberry-detection-counting/src/blueberry-detection-benchmark-firebase-adminsdk-khx5o-41dd2be9b2.json'
    cred = credentials.Certificate(service_account_json_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    storage_client = storage.Client.from_service_account_json(service_account_json_path)

    collection_name = "yolov8_metrics"
    all_documents = read_all_documents_from_collection(db, collection_name)
    print(all_documents)

    table_data = []
    for element in all_documents:
        row = {
            "model_type": element['methadata'].get('model_type', None),
            "epochs": element['methadata']['run_params'].get('epochs', None),
            "number_params": element['methadata'].get('total_params', None),
            "map_50": element.get('map_50', None),
            "map_50:95": element.get('map_50:95', None),
            "f1_50": element.get('f1_50', None),
            "p_50": element.get('p_50', None),
            "r_50": element.get('r_50', None),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    df["model_type"] = df["model_type"].str.replace(".pt", "", regex=False)

    df["number_params"] = (df["number_params"] / 1_000_000).round(3)

    numeric_columns = ["map_50", "map_50:95", "f1_50", "p_50", "r_50"]
    df[numeric_columns] = df[numeric_columns].round(4)
    df = df.sort_values(by="map_50", ascending=False)

    custom_column_names = {
        "model_type": "Model Type",
        "epochs": "Training Epochs",
        "number_params": "Parameters (Millions)",
        "map_50": "mAP@0.5",
        "map_50:95": "mAP@0.5:0.95",
        "f1_50": "F1 Score@0.5",
        "p_50": "Precision@0.5",
        "r_50": "Recall@0.5"
    }

    df_display = df.rename(columns=custom_column_names)

    display_table_with_styled_headers(df_display)


    return



#
# if __name__ == "__main__":
#
#     service_account_json_path = '/home/pqbas/labinm/PE501086701-2024-PROCIENCIA/blueberry-detection-counting/src/blueberry-detection-benchmark-firebase-adminsdk-khx5o-41dd2be9b2.json'
#     cred = credentials.Certificate(service_account_json_path)
#     if not firebase_admin._apps:
#         firebase_admin.initialize_app(cred)
#     db = firestore.client()
#     storage_client = storage.Client.from_service_account_json(service_account_json_path)
#
#     collection_name = "yolov8_metrics"
#     all_documents = read_all_documents_from_collection(db, collection_name)
#     print(all_documents)
#
#     table_data = []
#     for element in all_documents:
#         row = {
#             "model_type": element['methadata'].get('model_type', None),
#             "epochs": element['methadata']['run_params'].get('epochs', None),
#             "number_params": element['methadata'].get('total_params', None),
#             "map_50": element.get('map_50', None),
#             "map_50:95": element.get('map_50:95', None),
#             "f1_50": element.get('f1_50', None),
#             "p_50": element.get('p_50', None),
#             "r_50": element.get('r_50', None),
#         }
#         table_data.append(row)
#
#     df = pd.DataFrame(table_data)
#
#     df["model_type"] = df["model_type"].str.replace(".pt", "", regex=False)
#
#     df["number_params"] = (df["number_params"] / 1_000_000).round(3)
#
#     numeric_columns = ["map_50", "map_50:95", "f1_50", "p_50", "r_50"]
#     df[numeric_columns] = df[numeric_columns].round(4)
#     df = df.sort_values(by="map_50", ascending=False)
#
#     custom_column_names = {
#         "model_type": "Model Type",
#         "epochs": "Training Epochs",
#         "number_params": "Parameters (Millions)",
#         "map_50": "mAP@0.5",
#         "map_50:95": "mAP@0.5:0.95",
#         "f1_50": "F1 Score@0.5",
#         "p_50": "Precision@0.5",
#         "r_50": "Recall@0.5"
#     }
#
#     df_display = df.rename(columns=custom_column_names)
#
#     display_table_with_styled_headers(df_display)
#
