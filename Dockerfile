FROM python:3.12
EXPOSE 8083
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port=8083", "--server.address=0.0.0.0" ]