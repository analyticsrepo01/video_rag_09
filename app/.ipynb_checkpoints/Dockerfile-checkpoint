FROM python:3.10
EXPOSE 8081
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8081", "--server.address=0.0.0.0"]
