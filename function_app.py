import azure.functions as func
import numpy as np
import pywt
from scipy.fft import fft, fftfreq
import scipy.signal
from scipy.signal import butter, lfilter
import json
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import io
import os
import json
from dotenv import load_dotenv
import requests
# from azureml.core import Workspace, Model
# from azureml.core.webservice import AciWebservice, Webservice
# from azureml.core.model import InferenceConfig
# from azureml.core.environment import Environment
# from azureml.core.conda_dependencies import CondaDependencies

load_dotenv()

# get the record state from the table
# to determine if we need to record new data or use the model to predict the collected data
def get_env_variables():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    table_service_client = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service_client.get_table_client("envVariables")
    try:
        entity = table_client.get_entity(partition_key="enviroment", row_key="variables")
        return entity['Flag'], entity.get('current_label', 'default_label'), entity.get('data_count', 0)
    except Exception as e:
        print("Error reading state:", e)
        return None, None  # return None if there is an error

# read the mapping table to get the Blynk token and virtual pin
# the "prediction" gesture control which virtual pin?
def read_mapping_table(prediction):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    table_service_client = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service_client.get_table_client("envVariables")
    try:
        entity = table_client.get_entity(partition_key="map", row_key=prediction)
        return entity['blynkToken'], entity['virtualPin']
    except Exception as e:
        print("Error reading mapping table:", e)
        return None, None
    
# get the value of the virtual pin
def get_virtual_pin_status(token, pin):
    url = f"https://blynk.cloud/external/api/get?token={token}&V{pin}"
    response = requests.get(url)
    if response.status_code == 200:
        return int(response.text)
    else:
        print(f"Failed to get the value of {pin}. Status Code: {response.status_code}")
        return None

# update the value of the virtual pin
def update_virtual_pin_status(token, pin, value):
    url = f"https://blynk.cloud/external/api/update?token={token}&V{pin}={value}"
    response = requests.get(url)
    # print the url
    print(f"update url: {url}")
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to update the value of {pin}. Status Code: {response.status_code}")
        return None

# update the record state in the table
def update_env_variables(record_state, label, data_count):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    table_service_client = TableServiceClient.from_connection_string(connection_string)
    table_client = table_service_client.get_table_client("envVariables")
    entity = {
        "PartitionKey": "enviroment",
        "RowKey": "variables",
        "Flag": record_state,
        "current_label": label,
        "data_count": data_count
    }
    table_client.upsert_entity(entity)    


# push new data to the blob
def upload_data_to_blob(data, label):
    '''
    data: list, the data to be uploaded
    label: str, the label of the data, also will be the name of the json file
    '''
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "omnisurface-ml-data"
    # initialize the blob service client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # generate the blob name
    blob_name = f"{label}.json"
    blob_client = container_client.get_blob_client(blob_name)
    
    # check if the blob exists
    # 🌟 current logic: if there is a "tap.json" file, and users add another bunch of tap data, the data will append to the existing "tap.json" file
    try:
        blob_client.get_blob_properties()
        # if the blob exists, download the data and append the new data
        stream = blob_client.download_blob()
        existing_data = json.loads(stream.readall())
        existing_data.append(data)
        new_data = json.dumps(existing_data)
    except Exception as e:
        # if the blob does not exist, create a new data file
        new_data = json.dumps([data]) # convert the numpy array to a list

    # upload the new data to the blob
    blob_client.upload_blob(new_data, overwrite=True)

# denoise function
def wavelet_denoise(data, wavelet, level):
    # signal decomposition using wavelet
    coeff = pywt.wavedec(data, wavelet, mode='per', level=level)
    
    # calculate the threshold
    sigma = np.median(np.abs(coeff[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    # threshold the wavelet coefficients
    coeff[1:] = [pywt.threshold(i, value=threshold, mode='soft') for i in coeff[1:]]
    
    # signal reconstruction using wavelet
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode='per')
    return reconstructed_signal

def fft_features(fft_results):
    # Assume fft_results is an array of FFT coefficients
    magnitude_spectrum = np.abs(fft_results)
    
    # Spectral Centroid： Indicates the "center of mass" of the spectrum, giving an idea of the "brightness" of a sound.
    spectral_centroid = np.sum(magnitude_spectrum * np.arange(len(magnitude_spectrum))) / np.sum(magnitude_spectrum)
    
    # Spectral Rolloff：The frequency below which a certain percentage of the total spectral energy (commonly 85% or 95%) is contained, which helps in differentiating between harmonic content and noise.
    spectral_rolloff_threshold = 0.85 * np.sum(magnitude_spectrum)
    cumulative_sum = np.cumsum(magnitude_spectrum)
    spectral_rolloff = np.where(cumulative_sum >= spectral_rolloff_threshold)[0][0]
    
    # Spectral Flux：Measures the amount of local spectral change between successive frames, useful for detecting events.
    spectral_flux = np.sum((np.diff(magnitude_spectrum) ** 2))
    
    # Total Spectral Energy：Sum of squares of the FFT coefficients can serve as a measure of the overall signal energy.
    total_spectral_energy = np.sum(magnitude_spectrum ** 2)
    
    return {
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flux': spectral_flux,
        'total_spectral_energy': total_spectral_energy
    }

def wavelet_features(signal):
    wavelet_transform = scipy.signal.cwt(signal, scipy.signal.ricker, widths=np.arange(1, 31))
    wavelet_energy = np.sum(wavelet_transform**2, axis=0)
    
    features = {
        "wavelet_energy": wavelet_energy
    }
    return features

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# use the stored blob data to train the model
# first download the data from the blob, preprocess the data (scale the data), and then train the model
# finally, save the model and scaler to the blob
@app.function_name("train_model")
@app.route(route="train_model")
def train_model(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # 设置Blob服务连接和容器
        blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        data_container_name = "omnisurface-ml-data"

        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)

        # machine learning data container
        data_container_client = blob_service_client.get_container_client(data_container_name)
        
        # 初始化存储特征和标签的列表
        features = []
        labels = []

        # 列出容器中的所有blobs
        data_blob_list = data_container_client.list_blobs()
        for blob in data_blob_list:
            # 获取blob客户端，用于读取blob内容
            blob_client = data_container_client.get_blob_client(blob)
            
            # 下载blob内容
            blob_data = blob_client.download_blob().readall()
            
            # 解析blob内容（假设内容是JSON格式的向量列表）
            data = json.loads(blob_data)
            
            # 将数据添加到特征列表中，文件名（去除.json后缀）作为标签
            label = blob.name.split('.')[0]  # 假设文件名形式如 "tap.json"
            features.extend(data)
            labels.extend([label] * len(data))  # 每个向量的标签都是当前文件的名称

        # 将特征列表转换为numpy数组
        features_array = np.array(features)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.25, random_state=42)

        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 计算测试集上的准确率
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 序列化模型和scaler
        model_bytes = io.BytesIO()
        scaler_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        joblib.dump(scaler, scaler_bytes)
        model_bytes.seek(0)
        scaler_bytes.seek(0)

        # machine learning model container
        model_container_name = "omnisurface-ml-model"
        model_container_client = blob_service_client.get_container_client(model_container_name)

        # 上传模型和scaler到 Blob Storage
        model_blob_client = model_container_client.get_blob_client("omnisurface_rf_model.pkl")
        scaler_blob_client = model_container_client.get_blob_client("omnisurface_rf_scaler.pkl")
        model_blob_client.upload_blob(model_bytes, overwrite=True)
        scaler_blob_client.upload_blob(scaler_bytes, overwrite=True)

        # save the model report to the blob, e.g. accuracy
        model_report = {
            "model_name": "Random Forest",
            "accuracy": accuracy
        }
        model_report_blob_client = model_container_client.get_blob_client("model_report.json")
        model_report_blob_client.upload_blob(json.dumps(model_report), overwrite=True)

        # return func.HttpResponse(f"Success. Model trained successfully with {len(X_train)} training samples and {len(X_test)} test samples. Test accuracy: {accuracy:.2f}", status_code=200)
        return func.HttpResponse(f"Success. Accuracy: {accuracy:.2f}", status_code=200)
    
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
    
def download_blob_to_stream(blob_service_client, container_name, blob_name):
    """Helper function to download blob content to a stream."""
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    return io.BytesIO(blob_data)

# get the model from the blob, load the model and scaler, and use the model to predict the data
def predict_data(combined_feature):
    try:
        # Azure Blob Storage 设置
        connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        model_container_name = "omnisurface-ml-model"
        model_blob_name = "omnisurface_rf_model.pkl"
        scaler_blob_name = "omnisurface_rf_scaler.pkl"

        blob_service_client = BlobServiceClient.from_connection_string(connect_str)

        # 下载模型和scaler
        model_stream = download_blob_to_stream(blob_service_client, model_container_name, model_blob_name)
        scaler_stream = download_blob_to_stream(blob_service_client, model_container_name, scaler_blob_name)

        # 加载模型和scaler
        model = joblib.load(model_stream)
        scaler = joblib.load(scaler_stream)

        # 数据预处理
        combined_feature_scaled = scaler.transform([combined_feature])  # 假设combined_feature是单个样本

        # 使用模型预测
        # prediction = model.predict(combined_feature_scaled)
        # return prediction[0]  # 返回预测结果
        threshold = 0
        probas = model.predict_proba(combined_feature_scaled)[0]

        max_proba = np.max(probas) # max_proba = np.max(probas, axis=0)
        prediction = model.predict(combined_feature_scaled)

        # map label to probability
        label_proba_dict = dict(zip(model.classes_, probas))

        print(f"Predictions: {prediction}, Label-Proba Dict: {label_proba_dict}")


        if max_proba >= threshold:
            return prediction[0], label_proba_dict
        else:
            return "None", label_proba_dict

        # results = []
        # for pred, max_proba, proba in zip(predictions, max_probas, probas):
        #     if max_proba >= threshold:
        #         result = (pred, max_proba, proba)
        #     else:
        #         result = ("None", max_proba, proba)
        #     results.append(result)
        #     print(f"Prediction: {result[0]}, Max Proba: {result[1]:.4f}, All Probas: {result[2]}")
        # return [result[0] for result in results]


    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# get the raw data from the ESP32, extract FFT features, and
# A: store the data in the blob (Azure storage) if the record state is True
# B: call the model to predict the data if the record state is False
# req json format: {"piezo": [1332, 2233, 3231, ...], "mic": [1234, 2452, 3133, ...]}
@app.function_name("esp_rawdata_process")
@app.route(route="esp_rawdata_process")
def esp_rawdata_process(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # get the JSON data from the request
        req_body = req.get_json()

        # get the data from the request
        piezo_raw = req_body.get('piezo', [])
        mic_raw = req_body.get('mic', [])

        print(f"Raw piezo length: {len(piezo_raw)}, Raw mic length: {len(mic_raw)}")

        # the target length for both data is 3000 (~0.5s)
        target_length = 3000
        if len(piezo_raw) < target_length:
            # repeat the final value to fill the array
            piezo_raw.extend([piezo_raw[-1]] * (target_length - len(piezo_raw)))
        elif len(piezo_raw) > target_length:
            # truncate the array
            piezo_raw = piezo_raw[:target_length]
        if len(mic_raw) < target_length:
            # repeat the final value to fill the array
            mic_raw.extend([mic_raw[-1]] * (target_length - len(mic_raw)))
        elif len(mic_raw) > target_length:
            # truncate the array
            mic_raw = mic_raw[:target_length]

        # should have the same length of 3000 for both arrays
       
        # convert the data to numpy arrays
        piezo_npArr = np.array(piezo_raw)
        mic_npArr = np.array(mic_raw)

        # Debug: Check for NaN values in numpy arrays
        if np.isnan(piezo_npArr).any() or np.isnan(mic_npArr).any():
            print("NaN values found in raw data arrays")
            return func.HttpResponse(
                "Invalid data: NaN values present",
                status_code=400
            )
        
        print(f"piezo_npArr: {piezo_npArr[:100]}, mic_npArr: {mic_npArr[:50]}")
        

        # denoise the piezo and mic data
        piezo_denoised = wavelet_denoise(piezo_npArr, 'db2', 2) # 这里使用了Daubechies小波（'db1'），也就是Haar小波，分解层数为1。
        mic_denoised = wavelet_denoise(mic_npArr, 'db2', 2)
        
        # Debug: Check denoised data
        print(f"Denoised piezo: {piezo_denoised[:100]}, Denoised mic: {mic_denoised[:50]}")

        # 检查降噪数据中的无限值
        if np.isinf(piezo_denoised).any() or np.isinf(mic_denoised).any():
            print("Infinite values found in denoised data arrays")
            return func.HttpResponse(
                "Invalid data: Infinite values present in denoised data",
                status_code=400
            )
        
        # fft the piezo and mic data
        piezo_fft = np.fft.fft(piezo_denoised)
        mic_fft = np.fft.fft(mic_denoised)

        # debug: check the fft results
        print(f"FFT piezo: {piezo_fft[:100]}, FFT mic: {mic_fft[:10]}")

        # Check for NaN values in FFT results
        if np.isnan(piezo_fft).any() or np.isnan(mic_fft).any():
            print("NaN values found in FFT results")
            return func.HttpResponse(
                "Invalid data: NaN values present in FFT results",
                status_code=400
            )

        # extract features from the fft results
        # piezo_spectral_feature = fft_features(piezo_fft)
        # piezo_spectral_feature = np.array(list(piezo_spectral_feature.values()))
        # piezo_wavelet_feature = wavelet_features(piezo_denoised)['wavelet_energy']
        # piezo_feature = np.concatenate((piezo_spectral_feature, piezo_wavelet_feature))
        piezo_feature = np.abs(piezo_fft)

        # mic_spectral_feature = fft_features(mic_fft)
        # mic_spectral_feature = np.array(list(mic_spectral_feature.values()))
        # mic_wavelet_feature = wavelet_features(mic_denoised)['wavelet_energy']
        # mic_feature = np.concatenate((mic_spectral_feature, mic_wavelet_feature))
        mic_feature = np.abs(mic_fft)

        print(f"FFT piezo feature: {piezo_feature[:10]}, FFT mic feature: {mic_feature[:10]}")

        # combine the features
        combined_feature = np.concatenate((piezo_feature, mic_feature))

        print(f"Combined feature: {combined_feature[:10]}")

        # get the record state
        record_state, label, data_count = get_env_variables()
        if record_state: # if the record state is True, store the data in the blob. The JSON data is for later model training.
            upload_data_to_blob(combined_feature.tolist(), label)
            update_env_variables(record_state, label, data_count + 1)
             # return the features as a JSON response
            return func.HttpResponse(
                body=json.dumps({"features": combined_feature.tolist()}),
                status_code=200,
                mimetype="application/json"
            )
        else: # if the record state is False, call the model to predict the data
            prediction, label_proba_dict = predict_data(combined_feature)

            # Read the mapping table to get Blynk token and virtual pin
            blynkToken, virtualPin = read_mapping_table(prediction)

            if blynkToken and virtualPin:
                # Get the current status of the virtual pin
                current_status = get_virtual_pin_status(blynkToken, virtualPin)

                if current_status is not None:
                    # Determine the new value based on the current status
                    new_value = 0 if current_status else 255
                    # Update the virtual pin status
                    update_virtual_pin_status(blynkToken, virtualPin, new_value)
            
            # print the prediction
            print(f"Prediction: {prediction}")
            
            return func.HttpResponse(
                body=json.dumps({"prediction": prediction, "label_proba_dict": label_proba_dict}),
                status_code=200,
                mimetype="application/json"
            )
    
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON",
            status_code=400
        )