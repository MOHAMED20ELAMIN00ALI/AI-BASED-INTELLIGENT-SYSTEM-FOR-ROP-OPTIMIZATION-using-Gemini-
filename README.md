# 🤖 AI-Based Intelligent System for ROP Optimization (Gemini-Enhanced)

Welcome to the official repository for the **Drilling Performance Optimization Simulator**. This project represents a modern approach to Petroleum Engineering, integrating the classical **Galle-Woods** analytical model with **Generative AI (Google Gemini)** for intelligent decision support.

## 🌟 Project Overview
This system is designed to optimize the **Rate of Penetration (ROP)** by analyzing real-time drilling parameters (WOB, RPM, Depth, and Time). The integration of **Gemini** transforms the simulator from a mere calculator into an "Intelligent Advisor" that provides engineering recommendations based on data trends.

## 🧠 Why Gemini?
In this specific version of the project, **Google Gemini** is utilized to:
1. **Intelligent Diagnostics:** Interpret data cleaning reports and suggest why certain outliers were removed.
2. **Dynamic Recommendations:** Analyze the optimized WOB and RPM to provide safety warnings and drilling efficiency tips.
3. **Code Optimization:** Refine the mathematical logic for faster computational performance during Monte Carlo simulations.

## 📂 Repository Structure
* 📁 `project/`: The core Python source code.
* 📁 `data/`: Sample drilling datasets used for model calibration.
* 📄 `gemini_logic.py`: The integration script between the simulator and Gemini API.
* 📄 `requirements.txt`: List of dependencies (CustomTkinter, Pandas, GenerativeAI).

## 🛠️ Technical Workflow
1. **Data Acquisition:** Importing raw field data via Excel.
2. **Intelligent Cleaning:** Using IQR and Gemini logic to filter sensor noise.
3. **Analytical Modeling:** Applying the Galle-Woods ROP equation.
4. **Stochastic Optimization:** Running Monte Carlo simulations to find the best drilling parameters.
5. **AI Analysis:** Generating a final report using the Gemini LLM.

## 🚀 Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Add your `GEMINI_API_KEY` in the environment variables.
4. Run `main.py` to start the Intelligent Simulator.

## 🎓 Academic Affiliation
* **Researcher:** Mohamed Elamin Ali
* **Institution:** Sudan University of Science and Technology (SUST)
* **Department:** Petroleum Engineering (Southern Campus)
* **Project Date:** 2026

---
*Developed with the assistance of Google Gemini to bridge the gap between AI and Oil & Gas Engineering.*
