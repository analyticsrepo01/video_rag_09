import streamlit as st
import pandas as pd
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import vertexai
from vertexai.language_models import TextEmbeddingModel

# Define constants
PROJECT_ID = "my-project-0004-346516"
LOCATION = "us-central1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
SHOTS_FILE_PATH = "./shots.csv"
CLIPS_DIR = ""



def list_deployed_indexes():
    """List all deployed indexes in Vertex AI Matching Engine."""
    client = aiplatform.gapic.IndexEndpointServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    index_list = []

    # List all index endpoints
    for index_endpoint in client.list_index_endpoints(parent=parent):
        for deployed_index in index_endpoint.deployed_indexes:
            index_list.append({
                "index_endpoint_name": index_endpoint.name,
                "deployed_index_id": deployed_index.id,
                "display_name": index_endpoint.display_name
            })

    return index_list

def get_embeddings(query, model):
    """Generate embeddings for the query using the specified model."""
    embeddings = model.get_embeddings([query])
    return embeddings[0].values

def load_shots_df(file_path):
    """Load shots_df from a local CSV file."""
    return pd.read_csv(file_path)

vertexai.init(project=PROJECT_ID, location=LOCATION)
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")

st.title("RAG Demo Search Application")

if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = None

shots_df = load_shots_df(SHOTS_FILE_PATH)
if shots_df.empty:
    st.write('saurabh shots_df', shots_df)

tab1, tab2 = st.tabs(["Select Index", "Query Clips"])

with tab1:
    st.header("Select a Deployed Index")
    deployed_indexes = list_deployed_indexes()
    if deployed_indexes:
        index_options = [f"{index['display_name']} ({index['deployed_index_id']})" for index in deployed_indexes]
        selected_option = st.selectbox("Choose an index to query from:", index_options)
        if selected_option:
            selected_index = next(index for index in deployed_indexes if f"{index['display_name']} ({index['deployed_index_id']})" == selected_option)
            st.session_state['selected_index'] = selected_index
            st.success(f"Selected index: {selected_index['display_name']} with ID {selected_index['deployed_index_id']}")
    else:
        st.warning("No deployed indexes found.")

with tab2:
    vertexai.init(project=PROJECT_ID, location="us-central1")
    if not st.session_state.get('selected_index'):
        st.warning("Please select an index first.")
    else:
        st.header("Query Video Clips")
        query = st.text_input("Enter your query:")
        
        if not query:
            st.write('saurabh query', query)
        if query:
            try:
                query_embedding = get_embeddings(query, text_embedding_model)

                selected_index = st.session_state['selected_index']
                deployed_index_id = selected_index['deployed_index_id']
                index_endpoint_name = selected_index['index_endpoint_name']
                
                if not selected_index or not deployed_index_id:                
                    st.write('saurabh selected_index', selected_index)
                    st.write('saurabh deployed_index_id', deployed_index_id)
                    st.write('saurabh index_endpoint_name', index_endpoint_name)

                my_index_id = "8368299436119425024"
                my_index_endpoint_id = "8281781065152987136"

                my_index = aiplatform.MatchingEngineIndex(my_index_id)
                my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(my_index_endpoint_id)
                
                my_index_name = my_index._gca_resource.name
                my_index_display_name = my_index.display_name
                my_index_id = my_index.name.split('/')[-1]

                my_index_endpoint_name = my_index_endpoint._gca_resource.name
                my_index_endpoint_display_name = my_index_endpoint.display_name
                my_index_endpoint_id = my_index_endpoint.name.split('/')[-1]
                my_index_endpoint_public_domain = my_index_endpoint.public_endpoint_domain_name

                my_index = aiplatform.MatchingEngineIndex(my_index_name)

                my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(my_index_endpoint_id)
                
                API_ENDPOINT=my_index_endpoint_public_domain
                INDEX_ENDPOINT=my_index_endpoint_name
                
                # Build FindNeighborsRequest
                datapoint = aiplatform_v1.IndexDatapoint(
                    feature_vector=query_embedding
                )

                find_neighbors_query = aiplatform_v1.FindNeighborsRequest.Query(
                    datapoint=datapoint,
                    neighbor_count=3
                )

                find_neighbors_request = aiplatform_v1.FindNeighborsRequest(
                    index_endpoint=index_endpoint_name,
                    deployed_index_id=deployed_index_id,
                    queries=[find_neighbors_query],
                    return_full_datapoint=False
                )
                
                if not API_ENDPOINT:                
                    st.write('saurabh API_ENDPOINT /n', API_ENDPOINT)
                # Initialize MatchServiceClient
                match_service_client = aiplatform_v1.MatchServiceClient(
                    client_options={"api_endpoint": API_ENDPOINT}
                )

                # st.write('saurabh match_service_client ', match_service_client)                
                # st.write("saurabh find_neighbors_request  /n",find_neighbors_request)                

                # Execute the request
                response = match_service_client.find_neighbors(find_neighbors_request)

                if not response:                                
                    st.write("saurabh response",response.nearest_neighbors)
                
                # Prepare a DataFrame to store results
                results = []

                for result in response.nearest_neighbors:
                    for neighbor in result.neighbors:
                        if not neighbor :
                            st.write("saurabh neighbor",neighbor)                        
                        clip_id = int(neighbor.datapoint.datapoint_id)
                        distance = neighbor.distance
                        df_match = shots_df.loc[shots_df.index == clip_id]
                        if not df_match.empty:
                            match_info = df_match.iloc[0].to_dict()
                            match_info['distance'] = distance
                            results.append(match_info)

                # Convert results to DataFrame
                df_new = pd.DataFrame(results)
                
                if not results:                                                
                    st.write('saurabh df_new', results , df_new)

                # Sort by distance
                df_sorted = df_new.sort_values(by="distance", ascending=True)
                st.write("Matching clips:")
                st.dataframe(df_sorted[["clip_name", "description", "distance"]])

                # Display each video with label
                for index, row in df_sorted.iterrows():
                    st.write(f"**Clip Name:** {row['clip_name']}")
                    video_path = CLIPS_DIR + row['clip_name']
                    st.video(video_path)

            except Exception as e:
                st.error(f"Error during query execution: {e}")
                st.error(f"API Endpoint: {API_ENDPOINT}")
                st.error(f"Index Endpoint: {index_endpoint_name}")
                st.error(f"Deployed Index ID: {deployed_index_id}")
