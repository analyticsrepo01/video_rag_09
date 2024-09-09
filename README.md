# video_rag_09
latest video rag using descriptions and comes with a UI and matching ending -- end to end

RAG Demo Search Application
This is a demo application for a Retrieval-Augmented Generation (RAG) search system using Google's Vertex AI Matching Engine. The app allows users to search for video clips based on text queries by embedding the text and comparing it to precomputed embeddings of the video clips.

Prerequisites
Before you begin, ensure you have the following installed:

Docker
Google Cloud SDK (gcloud) with appropriate access permissions
A Google Cloud project with Vertex AI enabled


Architecture Diagram



Setup

1. Clone the Repository

git clone https://github.com/your-repo/rag-demo-app.git
cd rag-demo-app



2. Update the Environment Variables
In the app.py file, ensure that the following environment variables are correctly set:

PROJECT_ID: Your Google Cloud project ID.
LOCATION: The region where your Vertex AI resources are deployed.
SHOTS_FILE_PATH: The path to your CSV file containing video metadata.
CLIPS_DIR: The directory where your video clips are stored.



3. Build and Run the Docker Container
To run the application using Docker:
Build the Docker image:

docker build -t rag-demo-app .


Run the Docker container:

docker run -p 8080:8080 rag-demo-app


This will start the Streamlit app, which can be accessed at http://localhost:8080 in your web browser.

4. Deploying on Google Cloud Run
To deploy the application on Google Cloud Run. Follow these steps:
Build and push the Docker image to Google Container Registry (GCR):

docker build -t gcr.io/YOUR_PROJECT_ID/rag-demo-app .
docker push gcr.io/YOUR_PROJECT_ID/rag-demo-app


Deploy the image to Google Cloud Run:

gcloud run deploy rag-demo-app \
  --image gcr.io/YOUR_PROJECT_ID/rag-demo-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated


After deployment, you'll receive a URL where your application is hosted.
Using the Application
Once the application is running, you can use it as follows:
Select a Deployed Index:
Navigate to the "Select Index" tab.
Choose one of the available deployed indexes from the dropdown list.
Query Video Clips:
Switch to the "Query Clips" tab.
Enter your search query into the text box.
The app will generate embeddings for your query and find the most relevant video clips.