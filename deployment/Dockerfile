FROM python:3.12-slim

WORKDIR /app

# When pushing to HF Spaces, files are often placed in the root of the repo
# We adjust the COPY commands to look for files in the current build context
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY models/best_model.joblib /app/models/best_model.joblib
COPY models/model_metadata.json /app/models/model_metadata.json

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
