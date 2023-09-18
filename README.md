# DATAFEST-HACKATHON-DASE1.0
<!-- Your existing image tag![download](https://github.com/reedington/DATAFEST_HACKATHON_DASE1.0/blob/main/assets/110972011/ed966804-927a-4554-aa49-7c8bbfa403f3/download.png) -->
<img src="https://github.com/reedington/DATAFEST-HACKATHON-DASE1.0/assets/110972011/13337d83-861f-4ad3-9502-95b42e1c07a7" width="500" height="500" />



🎯Project Brief
---
Fraud Detection for Online Payment Platform
Overview

The Fraud Detection dataset is a crucial asset for our business, providing valuable insights and opportunities for enhancing the security and trustworthiness of our online payment platform. This dataset represents transactions and user-related data collected over time from our platform. The primary goal is to develop an advanced predictive model to identify potentially fraudulent transactions.

Context

Our online payment platform processes millions of transactions daily, making it vulnerable to various types of fraudulent activities. These activities pose a significant threat to both our business and our customers. To safeguard our platform and enhance user experience, we aim to leverage the power of data science and machine learning to proactively detect and prevent fraudulent transactions.

Objectives
- Fraud Detection Model: You are required to build an advanced machine learning model to predict whether a given transaction is potentially fraudulent or not. This model will be vital for real-time decision-making, allowing us to flag and investigate suspicious activities promptly.

- Enhanced Security: The primary aim is to enhance the security of our platform. By identifying fraudulent transactions early, we can take preventive measures to protect our customers and our business from financial losses.

- User Trust: Fraud detection directly impacts the trust our customers have in our platform. Accurate and efficient detection of fraudulent activities assures our users that their transactions are safe and secure.

- Operational Efficiency: Implementing automated fraud detection reduces the manual effort required for monitoring transactions, allowing our business to operate more efficiently.

Business Impact
The success of this project has significant implications for our business. An effective fraud detection system will help us maintain the integrity of our platform, foster user trust, and ensure the sustainability of our operations. By reducing fraud-related losses and improving the overall user experience, we aim to achieve long-term growth and success in the competitive online payment industry.

Conclusion
The Fraud Detection dataset is a valuable asset in our ongoing efforts to combat fraudulent activities on our online payment platform. We look forward to the innovative solutions and insights that you will bring to the datathon, ultimately contributing to a safer and more reliable online payment experience for our customers.

🛠️ Tools & Resources
You are at liberty to use any tool of your choice.

---

# Fraud Detection Model using PySpark

## Overview

This repository contains a PySpark-based fraud detection model. The model processes a large dataset, transforms it into a suitable format, and trains a Gradient Boosted Tree (GBT) classifier to detect fraudulent transactions.

## Prerequisites

- [PySpark](https://spark.apache.org/docs/latest/api/python/getting_started/index.html) installed
- Java installed and configured (required for PySpark)
- Large dataset (e.g., "Fraud Detection Dataset.csv") for training and testing

## Execution

1. **Environment Setup**

   - Ensure that Java and PySpark are installed and properly configured.
   - Clone this repository to your local machine.

2. **Data Loading**

   - Load the large CSV file (e.g., "Fraud Detection Dataset.csv") into the appropriate directory.

3. **Configuration**

   - Modify the `spark_conf` and `batch_size` variables in the Python script to configure Spark and batch size.

4. **Run the Code**

   - Execute the Python script `fraud_detection.py` to start processing the data and training the GBT classifier.

5. **Monitoring Progress**

   - The code processes data in batches, and the progress can be monitored in the terminal. It may take several hours to complete.

6. **Performance**

   ![Screenshot (343)](https://github.com/reedington/DATAFEST-HACKATHON-DASE1.0/assets/110972011/8780e7dc-a607-476f-8033-fad082cdcf4d)

   - The model aims to achieve an accuracy of approximately 0.87% across 1200 batches. The performance of each batch is printed during execution.

8. **Results**

   - After completion, the code will print the average accuracy achieved across all batches.

9. **Clean-Up**

   - Ensure to stop the Spark session to release resources.

## Notes

- The code can be further customized by adjusting hyperparameters and model configurations.

## Author

[Adeleke Oluwafikayomi](https://www.linkedin.com/in/oluwafikayomi-adeleke-98a29023b/)

Feel free to reach out for any questions or assistance.

- The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1NDRx33ohBh3_LlVhygzc8n3Q3F8OE2Od).
