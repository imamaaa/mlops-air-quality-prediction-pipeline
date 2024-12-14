# Weather Monitoring System

## Project Overview
This project collects and tracks weather and pollution data for Islamabad using OpenWeatherMap API and implements MLOps practices with DVC.

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate virtual environment: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create `.env` file with OpenWeatherMap API key
6. Initialize DVC: `dvc init`
7. Configure DVC remote storage

## Data Collection
- Weather and pollution data collected every 4 hours
- Data stored in JSON format
- Version controlled using DVC
- Automated using Windows Task Scheduler

## Directory Structure
- `data/`: Stores collected weather and pollution data
- `scripts/`: Contains collection scripts and automation files
- `logs/`: Contains execution logs

## Task Scheduler Configuration
Collection times:
- 00:00, 04:00, 08:00, 12:00, 16:00, 20:00