import base64
import cv2
import numpy as np
import supabase

# Create a Supabase client instance
supabase_url = 'https://gkznooznskcndgxcecbs.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdrem5vb3puc2tjbmRneGNlY2JzIiwicm9sZSI6ImFub24iLCJpYXQiOjE2Nzc2MzU3NzMsImV4cCI6MTk5MzIxMTc3M30._G03-8_aaemIiZ-69PdEI4p485iu4OE8lU7HJBKFevE'
supabase_client = supabase.create_client(supabase_url, supabase_key)

# Initialize the Flask application
app = Flask(__name__)

# Enable CORS
CORS(app)

@app.route('/sift', methods=['POST'])
def sift():
    # Get the image data from the POST request
    image_data = request.get_json()['image']
    image = np.fromstring(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Apply the SIFT algorithm to the image and extract the descriptors
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)

    # Save the descriptors to Supabase
    descriptors_data = des.tobytes()
    response = supabase_client.storage.from_path('descriptors.npy').upload(descriptors_data)

    # Return a success message
    return jsonify({'message': 'Descriptors saved successfully'})


@app.route('/match', methods=['POST'])
def match():
    # Get the query image from the POST request
    query_image_data = request.get_json()['query_image']
    query_image = np.fromstring(base64.b64decode(query_image_data), np.uint8)
    query_image = cv2.imdecode(query_image, cv2.IMREAD_COLOR)

    # Get the descriptors from Supabase
    response = supabase_client.storage.from_path('descriptors.npy').download()
    descriptors_data = response.content
    descriptors = np.frombuffer(descriptors_data, np.float32)

    # Reshape the descriptors to the original shape
    num_descriptors = descriptors.shape[0] // 128
    descriptors = descriptors.reshape(num_descriptors, 128)

    # Apply the SIFT algorithm to the query image
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(query_image, None)

    # Create a Brute-Force matcher and match the descriptors
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors, des)

    # Calculate the similarity score as 100 times the number of matches divided by the minimum number of descriptors
    similarity_score = 100 * len(matches) / min(num_descriptors, des.shape[0])

    # Return the similarity score
    return jsonify({'similarity_score': similarity_score})

