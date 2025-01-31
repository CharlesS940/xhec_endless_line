# Endless Line Project Commercial Proposal with Eleven Strategy | X-HEC 2024-25

## Authors
Cathal Brady, Dora Bijvoet, Ameya Kulkarni, Charles Siret, Hadrien Strichard, Hocine Zidi

##

## Overview
This repository contains the notebook demonstrating the code for:
- **Predicting waiting times**
- **Calculating optimal scheduling**

It also includes a **Streamlit app** to showcase a demo of the solution.

> **Disclaimers:**
- This project is a part of a mock demo for a commercial proposal. It is not the final product and should not be treated as such.
- Due to size limits of files on github, some csv were stored using lfs, make sure to use their explicit version before running any code.

---

## 📂 Code Overview

### 1) Predicting Waiting Times  
- **preprocessing_and_catboost**

### 2) Optimal Scheduling  
- **app.py**: Code for the dynamic scheduling streamlit app
- **capacity_utilisation**: Determines Capacity and optimal pathing

### 3) Others  
- **ride_data**: Extracts details of rides such as capacity.  
- **synthetic_data_generation**: Creates synthetic user preference data for the app.  
- **map**: Develops a sample map of rides for optimal pathing.

---

## 🚀 How to Run the Streamlit Demo

1. **Create a Conda environment:**  
   ```bash
   conda create --name endless_line_env python=3.8
   conda activate endless_line_env
   ```

2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to the project's working directory:**  
   ```bash
   cd /path/to/project
   ```

4. **Run the Streamlit app:**  
   ```bash
   streamlit run app.py
   ```

5. **Open in browser:**  
   A browser window should open automatically. If not, navigate to:  
   👉 [http://localhost:8501](http://localhost:8501)

> **Note:**
If running the app causes any bugs related to missing data or key errors, please run the capacity_utilization notebook fully before running the app again.
---

### 🔧 Future Improvements
- Improve waiting time prediction with real-time updates.
- Enhance user interface for better experience.
- Optimize scheduling algorithms for larger datasets.

📬 Feel free to reach out if you have any questions!
