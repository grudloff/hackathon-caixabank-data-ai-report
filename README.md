# Hackathon Caixabank Data AI Report

This repository contains my submission for the **Caixabank Data AI Hackathon**. The project addresses various data science tasks aimed at analyzing financial transactions, detecting fraud, forecasting expenses, and developing an AI agent for report generation.

## 📋 Overview of Tasks

### **Task 1: Data Statistics**
- **Objective:** Answer specific queries related to card and client data.
- **Approach:** Implemented functions in `src/data/data_functions.py` to process the `transactions.csv` dataset, extracting required information such as the `card_id` with the latest expiry date and lowest credit limit, and identifying clients nearing retirement with specific financial metrics. This is straightforward demostration of data manipulation and aggregation techniques with Pandas.

### **Task 2: Data Functions Implementation**
- **Objective:** Develop functions to compute earnings, expenses, and cash flow summaries.
- **Approach:** Created functions like `earnings_and_expenses`, `expenses_summary`, and `cash_flow_summary` in `src/data/data_functions.py`, ensuring accurate data aggregation and transformation based on provided parameters. These functions implement data processing techniques to generate financial insights for clients.

### **Task 3: Fraud Detection Model**
- **Objective:** Build a model to classify transactions as fraudulent or non-fraudulent.
- **Approach:** Utilized supervised learning techniques with handling for class imbalance. The model was trained on labeled data and evaluated using the Balanced Accuracy score.
- **Model:** Utilized a Xgboost model with undersampling the majority class and weighted classes to handle dataset imbalance.

### **Task 4: Expenses Forecasting**
- **Objective:** Forecast future monthly expenses for each client.
- **Approach:** Employed time series forecasting models to predict expenses over the next three months using historical transaction data, ensuring personalized budget management insights.
- **Model:** Utilized a LSTM model to forecast future expenses based on historical transaction data. The model was implemented utilizin the `pytorch-forecast` library that implements appropriate dataloaders and models.

### **Task 5: AI Agent for Report Generation**
- **Objective:** Develop an AI agent capable of generating PDF reports based on natural language input.
- **Approach:** Leveraged `Langchain` and the `llama3.2:1b` language model hosted locally via Ollama. The agent processes user input to extract relevant dates and invokes functions to generate comprehensive PDF reports, as implemented in `src/agent/agent.py`. The approach ensures the agent can generate reports by utilizing a single tool, requiring from the agent that the dates are extracted from the input to call the tool. The `client_id` and dataframe are injected into the tool at runtime. To handle potential errors in date extraction by the model, the prompt specifies the number of days for each month of the year.

## 🗂️ Repository Structure

```
├── data/
│   ├── raw/
│   │   └── mcc_codes.json    
│   └── processed/              
│
├── predictions/   
│   ├── predictions_1.json 
│   ├── predictions_3.json 
│   └── predictions_4.json
│
├── src/                       
│   ├── data/                   
│   │   ├── api_calls.py       
│   │   ├── data_functions.py       
│   │   └── data_questions.py     
│   │
│   ├── models/                 
│   │   ├── train_model.py      
│   │   └── predict_model.py   
│   │
│   └── agent/                  
│       ├── agent.py            
│       └── tools.py            
│
├── tests/                      
│   ├── agent_test.py       
│   ├── statistics_test.py 
│   └── conftest.py                    
│
├── README.md  
└── requirements.txt     
```

## 🛠️ Getting Started

1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/hackathon-caixabank-data-ai-report.git
    ```
2. **Navigate to the Project Directory**
    ```bash
    cd hackathon-caixabank-data-ai-report
    ```
3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4. **Install the src package**
    ```bash
    pip install -e .
    ```
5. **Install Ollama**
    ```bash
    bash install_ollama.sh
    ```

## 📚 References

Original competition details can be found in the `COMPETITION.md` file.
