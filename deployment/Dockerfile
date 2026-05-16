FROM python:3.12-slim

WORKDIR /app

# When pushing to HF Spaces, files are often placed in the root of the repo
# We adjust the COPY commands to look for files in the current build context
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
