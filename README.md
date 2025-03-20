# MLOps Environmental Monitoring & Pollution Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MLOps](https://img.shields.io/badge/MLOps-DVC%2C%20MLflow%2C%20Docker-green)
![Monitoring](https://img.shields.io/badge/Monitoring-Grafana%2C%20Prometheus-orange)

## 📌 Project Overview
This project implements an **MLOps pipeline** for monitoring **air pollution** and predicting **Air Quality Index (AQI)** using machine learning. The system integrates **data collection, model development, deployment, and monitoring** into a streamlined workflow.  

### 🔹 Key Objectives
✅ Automate **data collection & versioning** using **DVC** 🔄  
✅ Develop a **time-series prediction model** (ARIMA/LSTM) for **AQI forecasting** 📈  
✅ Deploy the model as an **API** using **Flask/FastAPI** 🌐  
✅ Set up **monitoring & alerting** using **Prometheus & Grafana** 📊  

### 🔹 Key MLOps Tools Used
- **DVC (Data Version Control)** → Tracks & manages datasets via **Amazon S3**
- **Flask/FastAPI** → Serves model predictions as an **API**
- **Prometheus & Grafana** → Monitors model performance & system health
- **Docker & Docker Compose** → Containerized deployment for scalability

---

## 🏗️ System Architecture
```mermaid
graph TD
    A --> [Real-time Data Collection] -->|APIs| B[DVC Versioned Data]
    B --> C[Time-Series Model (ARIMA/LSTM)]
    C -->|Predictions| D[Flask/FastAPI API]
    D -->|Live Monitoring| E[Grafana & Prometheus]
```

---

## 🔥 Features Implemented
| Feature               | Description |
|-----------------------|-------------|
| ✅ **Data Collection** | Fetches real-time AQI & weather data via OpenWeather API 📡 |
| ✅ **Automated Scheduling** | **Windows Task Scheduler** executes the batch file every 4 hours for continuous data collection 🔄 |
| ✅ **Data Versioning** | Uses **DVC** with **Amazon S3** for dataset tracking 🛢️ |
| ✅ **Model Training** | Develops **ARIMA & LSTM** for AQI forecasting 📈 |
| ✅ **Model Deployment** | Deploys predictions via **Flask/FastAPI API** 🌐 |
| ✅ **Live Monitoring** | Uses **Grafana & Prometheus** for tracking performance 📊 |

---

## 🔄 Workflow
1️⃣ **Data Collection**: The system fetches real-time AQI & weather data from **OpenWeather API**. Data collection is automated using **Windows Task Scheduler**, which runs every 4 hours.
2️⃣ **Data Versioning**: The collected data is stored and tracked using **DVC (Data Version Control)** with **Amazon S3** as remote storage.
3️⃣ **Model Development**: The system trains **time-series models (ARIMA & LSTM)** to predict future air quality levels.
4️⃣ **Model Deployment**: The trained model is deployed as an API using **Flask/FastAPI**, allowing users to make real-time predictions.
5️⃣ **Monitoring & Logging**: The deployed API and model performance are continuously monitored using **Prometheus & Grafana**, providing real-time metrics and visualizations.

---

## 📜 API Endpoints
| Method | Endpoint           | Description |
|--------|-------------------|-------------|
| GET    | `/predict`         | Get pollution level prediction |
| POST   | `/predict`         | Send input data for model inference |
| GET    | `/metrics`         | API & model performance metrics |

---

## 📊 Monitoring Setup (Grafana & Prometheus)

1️⃣ Start **Prometheus & Grafana** using Docker Compose:
```bash
docker-compose up -d
```
2️⃣ Access **Grafana Dashboard** → `http://localhost:3000`

---

## 🔧 Challenges & Key Learnings

### 🚧 Challenges Faced
- **Real-time data ingestion** while ensuring dataset versioning with **DVC**  
- **Optimizing time-series models** (ARIMA/LSTM) for air quality forecasting  
- **Setting up Prometheus & Grafana** for real-time monitoring  

### 🎯 Key Learnings
- **Automating data pipelines** with **DVC** & **Amazon S3**  
- **Deploying ML models as APIs** with **Flask/FastAPI**  
- **Monitoring ML systems** with **Prometheus & Grafana**  

---

## 🚀 Future Improvements
✅ **Implement experiment tracking** using **MLflow**  
✅ **Optimize model inference time** for real-time predictions  
✅ **Deploy to cloud-based services** (AWS, GCP, Azure)  
✅ **Integrate alerting system** for high pollution days  

---

## 📦 Deployment on Docker Hub

If you want to **publish the Flask application as a Docker image**, follow these steps:

1️⃣ **Build the Docker image:**
```bash
docker build -t YOUR_DOCKERHUB_USERNAME/mlops-aqi .
```
2️⃣ **Push the image to Docker Hub:**
```bash
docker push YOUR_DOCKERHUB_USERNAME/mlops-aqi
```
Now, anyone can pull and run your application with:
```bash
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/mlops-aqi
```

---

## 🤝 Contributing
Pull requests are welcome! If you have suggestions, feel free to open an issue.

---

## 📄 License
This project is licensed under the **MIT License**.

---

## ⭐ Like This Project?
Give it a ⭐ on GitHub and connect with me on [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)!

---

Now your project is **recruiter-friendly, well-documented, and optimized for GitHub!** 🚀🔥 Let me know if you need any refinements! 😊

