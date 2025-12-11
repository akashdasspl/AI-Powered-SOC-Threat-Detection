# Machine Learning Intrusion Detection System (IDS)

## Overview
This repository contains a full SOC‑grade Intrusion Detection System integrating:
- FastAPI ML model server
- Logstash enrichment pipeline
- Elasticsearch alert storage
- Kibana SOC dashboard

## Features
- Real‑time ML‑powered threat detection
- Automated enrichment and severity scoring
- Interactive SOC dashboard
- Simulated malicious traffic testing

## Architecture
1. Logs → Logstash  
2. Logstash → FastAPI ML Model  
3. ML Output → Logstash → Elasticsearch  
4. Visualization → Kibana Dashboard  

## Files Included
- `logstash.conf` — ML enrichment pipeline  
- `main.py` — FastAPI model server  
- `IDS_Project_Report.pdf` — Full project documentation  
- `IDS_Project_Presentation.pptx` — SOC presentation  
- Docker setup for reproducibility  

## How to Run
```
docker compose up --build
python ml/api/predict_api.py  


```

## Dashboard
http://localhost:5601/
fast Api
http://localhost:8000/docs#/default/predict_predict_post

Open Kibana → Dashboards → SOC Detection Dashboard.

## Author
Akash das,Akshay kumar banjare,Aman Parganiha ,Anurag kumar sahu ,Ayush Rajput
