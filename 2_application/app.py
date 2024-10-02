import os
import base64
import streamlit as st
import anthropic
from anthropic import Anthropic
import requests

# Initialize Claude client with the API key from environment variables
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Set up the Streamlit application with a title and layout
st.set_page_config(page_title="Image Analysis with Anthropic Claude", layout="wide")
st.title("Transcription and Information Extraction with Anthropic Claude")

# Function to encode an image to a base64 string
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        base64_string = base_64_encoded_data.decode('utf-8')
        return base64_string

# Function to filter and return only image files from a directory
def get_image_files(directory, extensions=("png", "jpg", "jpeg", "gif", "webp")):
    return [f for f in os.listdir(directory) if f.lower().endswith(extensions) and os.path.isfile(os.path.join(directory, f))]

# Display the default image with an option to enlarge and shrink
def display_image_with_enlarge_option(image_path, caption="Image"):
    # Toggle state for enlarging/shrinking the image
    key_prefix = caption.replace(" ", "_") + "_" + os.path.basename(image_path)
    
    # Initialize session state for image enlargement if not already present
    if f"enlarged_{key_prefix}" not in st.session_state:
        st.session_state[f"enlarged_{key_prefix}"] = False

    # Set image dimensions based on toggle state
    width = 200  # Small size
    enlarged_width = width * 3  # Tripled size

    # Display button to toggle enlarge/shrink
    if st.session_state[f"enlarged_{key_prefix}"]:
        st.image(image_path, caption=f"Enlarged {caption}", width=enlarged_width)
        if st.button("Shrink Image", key=f"shrink_{key_prefix}"):
            st.session_state[f"enlarged_{key_prefix}"] = False
    else:
        st.image(image_path, caption=f"Small {caption}", width=width)
        if st.button("Enlarge Image", key=f"enlarge_{key_prefix}"):
            st.session_state[f"enlarged_{key_prefix}"] = True

# Function to send a prompt to the Claude model
def send_claude_request(image_path, instruction, model):
    # Convert image to base64 string
    image_data_base64 = get_base64_encoded_image(image_path)
    
    HUMAN_PROMPT = "\n\nHuman:"
    AI_PROMPT = "\n\nAssistant:"
    
    # Format the prompt using HUMAN_PROMPT and AI_PROMPT
    prompt = f"{HUMAN_PROMPT} Here is an image data encoded in base64 format:\n{image_data_base64}\nPlease perform the following instruction: {instruction}{AI_PROMPT}"
    if image_path.endswith(".png"):
        media_type = "image/png"

    if image_path.endswith(".gif"):
        media_type = "image/gif"
        
    if image_path.endswith(".jpeg"):
        media_type = "image/jpeg"

    if image_path.endswith(".webp"):
        media_type = "image/webp"
    
    message_list = [
        {
            "role": 'user',
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": get_base64_encoded_image(image_path)}},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    # Send request to Claude model using the updated syntax
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=message_list
    )
    
    return response.content[0].text


# Function to dynamically fetch available models from the client
def get_available_models():
    return ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]

# Define the three tabs for the Streamlit app
tab1, tab2, tab3 = st.tabs(["Use Cases for Anthropic Claude Models", "Upload and View Images", "About"])

# Use Case Tab - Tab 1
with tab1:
    # Use two columns for layout: image on the left (1/3), response on the right (2/3)
    col1, col2 = st.columns([1, 2])
        
    with col1:
        st.header("Select Image Analysis Use Case")
        
        # Define possible use cases for image transcription
        use_cases = [
            "Transcribing Typed Text",
            "Transcribing Handwritten Text",
            "Transcribing Forms",
            "Complicated Document QA",
            "Unstructured Information -> JSON",
            "User Defined"
        ]
        
        # Dropdown menu for selecting a use case
        selected_use_case = st.selectbox("Choose a Use Case:", use_cases)
        
        # Instructions for each use case
        instructions = {
            "Transcribing Typed Text": "Transcribe this typed text.",
            "Transcribing Handwritten Text": "Transcribe this handwritten note.",
            "Transcribing Forms": "Transcribe this form exactly.",
            "Complicated Document QA": "Answer the questions based on this document.",
            "Unstructured Information -> JSON": "Convert the content of this document to structured JSON.",
            "User Defined": ""
        }

        # Descriptions for each use case
        use_case_descriptions = {
            "Transcribing Typed Text": "Extract text from typed documents like printouts or scanned PDFs.",
            "Transcribing Handwritten Text": "Extract text from handwritten notes, making them searchable or editable.",
            "Transcribing Forms": "Extract data from structured forms while preserving its organization.",
            "Complicated Document QA": "Answer questions based on the contents of a complex document.",
            "Unstructured Information -> JSON": "Convert unstructured document content into a structured JSON format.",
            "User Defined": "Create your own prompt for processing the image with a custom instruction."
        }

        # Display the description below the use case dropdown
        st.markdown(f"**Description:** {use_case_descriptions[selected_use_case]}")
        
        # Fetch available models dynamically and select one
        available_models = get_available_models()
        selected_model = st.selectbox("Choose a model:", available_models)
        
        # Initialize the base instruction for all use cases except "User Defined"
        if selected_use_case == "User Defined":
            instruction_text = st.text_area("Enter your prompt:", "")
        else:
            instruction_text = instructions[selected_use_case]
            
            # Check if additional context is needed for the "Complicated Document QA"
            if selected_use_case == "Complicated Document QA":
                additional_context = st.text_area("Enter question you would like to know about document:", "")
                
                # Append additional context for "Complicated Document QA" if provided
                if additional_context:
                    instruction_text += f" {additional_context}"
        
        # Option to use either an uploaded image or an example image, excluding "User Defined" case
        if selected_use_case != "User Defined":
            image_selection = st.radio("Choose an image:", ("Use uploaded image", "Use example image"))
        else:
            image_selection = "Use uploaded image"  # Force image selection to be only "Use uploaded image"

        # Variable to hold the selected image path
        image_path = None
        # Initialize session state for the selected image if not already present
        if 'selected_image' not in st.session_state:
            st.session_state['selected_image'] = None
        
        if image_selection == "Use uploaded image":
            # Directory containing uploaded images
            uploaded_images_dir = "/home/cdsw/data"
            
            # Get only image files from the directory
            images_list = get_image_files(uploaded_images_dir)
            
            # Let user select from uploaded images if available
            if images_list:
                image_name = st.selectbox("Select an uploaded image:", images_list)
                image_path = os.path.join(uploaded_images_dir, image_name)
            else:
                st.warning("No uploaded images found. Please upload an image in the 'Upload Image' tab.")
        else:
            # Placeholder for default images based on use case
            default_images = {
                "Transcribing Typed Text": "./data/examples/ex1-stack_overflow.png",
                "Transcribing Handwritten Text": "./data/examples/ex2-school_notes.png",
                "Transcribing Forms": "./data/examples/ex3-vehicle_form.jpeg",
                "Complicated Document QA": "./data/examples/ex4-doc_qa.jpeg",
                "Unstructured Information -> JSON": "./data/examples/ex5-org_chart.jpeg"
            }
            image_path = default_images.get(selected_use_case)
        
            # Display image with the option to enlarge and shrink
            if image_path:
                display_image_with_enlarge_option(image_path, caption="Selected Image")
        
        # Placeholder for response
        response=""
        
        # Process the request if an image and prompt are provided
        if image_path and instruction_text and st.button("Process with Claude"):
            # Send the request to Claude and display the response
            response = send_claude_request(image_path, instruction_text, selected_model)

            
    with col2:
        # Display the response in a well-formatted block
        st.header("Claude's Response:")
        st.code(response, language="markdown")

        
# Image Upload Tab - Tab 2
with tab2:
    st.header("Manage Uploaded Images")
    
    # Directory for saving uploaded images
    upload_directory = "/home/cdsw/data"
    os.makedirs(upload_directory, exist_ok=True)
    
    # Initialize session state to track uploaded file name
    if 'uploaded_file_name' not in st.session_state:
        st.session_state['uploaded_file_name'] = None

    # File uploader allowing only specific image types
    uploaded_file = st.file_uploader("Upload an image for transcription", type=["png", "jpg", "jpeg", "gif", "webp"])

    # Check if a file is uploaded and not already processed
    if uploaded_file and uploaded_file.name != st.session_state['uploaded_file_name']:
        # Save the uploaded image to the directory
        file_path = os.path.join(upload_directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Image saved to {file_path}")
        
        # Update session state with the current file name
        st.session_state['uploaded_file_name'] = uploaded_file.name
        
        # Trigger a UI refresh
        st.rerun()

    # Clear the file name once the upload is completed or no file is selected
    if not uploaded_file:
        st.session_state['uploaded_file_name'] = None

    # Create two columns: one for listing images and action buttons, and one for rendering selected image
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Uploaded Images")
        
        # Get a list of image files from the directory
        image_files = get_image_files(upload_directory)
        
        # Check if there are any images in the folder
        if image_files:
            # Store the selected image name in session state
            if 'selected_image' not in st.session_state:
                st.session_state['selected_image'] = None
            
            # Loop through each image and display its name with "View" and "Delete" buttons
            for image_file in image_files:
                image_path = os.path.join(upload_directory, image_file)
                
                # Create keys for each button
                view_button_key = f"view_{image_file}"
                delete_button_key = f"delete_{image_file}"
                
                col_img_name, col_view, col_delete = st.columns([3, 1, 1])
                
                with col_img_name:
                    st.text(image_file)
                
                with col_view:
                    # "View" button to display the image in the right column
                    if st.button("View", key=view_button_key):
                        st.session_state['selected_image'] = image_path
                
                with col_delete:
                    # "Delete" button to remove the image from the directory
                    if st.button("Delete", key=delete_button_key):
                        os.remove(image_path)
                        st.success(f"{image_file} has been deleted.")

                        # Clear the selected image to avoid trying to display a deleted file
                        if st.session_state.get('selected_image') == image_path:
                            st.session_state['selected_image'] = None

                        # Refresh the list of images after deletion
                        image_files = get_image_files(upload_directory)

                        # If no images are left, reset the selected image state
                        if not image_files:
                            st.session_state['selected_image'] = None

                        st.rerun()  # Refresh the UI after deletion


        else:
            st.info("No images found in the directory.")
    
    with col2:
        # Render the selected image in the right column
        if st.session_state['selected_image']:
            st.subheader("Selected Image")
            st.image(st.session_state['selected_image'], caption="Preview", use_column_width=True)


# About Tab - Tab 3
with tab3:
    st.header("About")

    # Use two columns to place text and GIF side by side
    col1, col2 = st.columns([3, 1])  # Adjust column width ratios as needed

    with col1:
        # Section 1: Original Overview of the App
        st.subheader("Exploring the Power of Claude")
        st.write(
            '''
            This application allows users to transcribe and extract information from various types of documents using the Claude model by Anthropic.
            The functionality includes transcribing typed or handwritten text, extracting data from forms, performing QA on complex documents, and converting unstructured information into JSON.
            '''
        )

        # Section 2: Advanced Capabilities
        st.subheader("Unlocking Claude's Potential")
        st.write(
            '''
            - **Advanced Reasoning**: Claude can perform complex cognitive tasks that go beyond simple pattern recognition or text generation.
            
            - **Vision Analysis**: Transcribe and analyze almost any static image, from handwritten notes and graphs to photographs.
            
            - **Code Generation**: Start creating websites in HTML and CSS, turning images into structured JSON data, or debugging complex code bases.
            
            - **Multilingual Processing**: Translate between various languages in real-time, practice grammar, or create multi-lingual content.
            '''
        )
        
        # Section 3: Model Selection Guide
        st.subheader("Choose Your Claude: A Model for Every Task")
        st.write(
            '''
            - **Light & Fast: Haiku**: Anthropic's fastest model that can execute lightweight actions, with industry-leading speed. Ideal for quick tasks where time is of the essence.
            
            - **Hard-working: Sonnet**: The best combination of performance and speed for efficient, high-throughput tasks. Strikes a balance between speed and power, making it suitable for most general-purpose tasks.
            
            - **Powerful: Opus**: Anthropic's highest-performing model, capable of handling complex analysis, longer tasks with many steps, and higher-order math and coding tasks. Best for situations where accuracy and depth are prioritized over speed.
            '''
        )

    with col2:
        # Display the GIF on the right side
        gif_path = "/home/cdsw/assets/claude.gif"  # Replace with the correct path to claude.gif
        st.image(gif_path, use_column_width=True)
