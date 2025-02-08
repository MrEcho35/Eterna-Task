# Conversion Prediction API 🚀

This API leverages the Prophet model to forecast conversion counts. 

## Key Features:
- Time-series forecasting using Prophet.
- FastAPI-based REST API for easy integration.
- Redis caching for faster predictions.
- Highly customizable and easy to deploy.

## Overview

This Conversion Prediction API provides accurate and efficient forecasting. Mainly implemented for 7 days forecasting but can be used for any amount of days.

## Requirements

To run this API, you'll need the following:

- **Python 3.8+**: Make sure you have Python 3.8 or above installed.
- **FastAPI**: A modern web framework for building APIs.
- **Prophet**: A tool for forecasting time-series data.
- **Redis**: Required for caching. In order to program to work Redis Server and Redis CLI must be running in the background. You can install Redis using the following link: [Redis for Windows](https://github.com/microsoftarchive/redis/releases).

## Installation Steps

### 1. Clone the repository:
```bash
git clone https://github.com/MrEcho35/Eterna-Task.git
cd Eterna-Task
```

### 2. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Open Redis Server and Redis CLI:
```bash
redis-server
```
Alternatively you can open them manually.

### Start the FastAPI:
```bash
uvicorn basic_api:app --reload
```

## Usage

- When app is opened click **API Documentation**.
- Click **POST /predict**.
- Click **Try it out** at the top right.
- Enter a number next to **days:** (preferably 7).
- Click **Execute** and check the response body.
