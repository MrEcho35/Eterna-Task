This is the repository of the task given by Eterna Technology.

# 🚀 Conversion Prediction API

This API makes conversion_count forecasting using Prophet model

## Requirements

- Python 3.8+
- FastAPI
- Prophet
- Redis (In order to program to work, Redis Server and Redis CLI must be running on the background. You can download from here: [Redis for Windows](https://github.com/microsoftarchive/redis/releases))

## 🚀 Installation
```bash
git clone https://github.com/MrEcho35/Eterna-Task.git
cd Eterna-Task
pip install -r requirements.txt (Download necessary libraries)
uvicorn basic_api:app --reload (Run program)
