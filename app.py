import waterpixels_tools as wpt
import streamlit as st
from PIL import Image
import numpy as np

# Define the transformation function
def apply_waterpixels_transform(image, k, sigma):
    # Apply your transformation logic here
    # For demonstration, let's convert the image to grayscale
    grayscale_image = image.convert("L")
    # Return the transformed image
    return grayscale_image

def main():
    st.title("Waterpixels - Image Viewer and Details App")  # Adding a big title
    
    # Input fields for user details
    k = st.number_input("Regularization Constant (k)", value=2, step=1)
    sigma = st.number_input("Step (sigma)", value=40, step=20)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the selected image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Details provided by the user:")
        st.write(f"- Regularization Constant (k): {k}")
        st.write(f"- Step (sigma): {sigma}")

        # Button to apply Waterpixels transformation
        if st.button("Waterpixels"):
            # Apply the transformation
            transformed_image = wpt.waterpixel(img_path=uploaded_file, k=k,step=sigma, plot=False)
            # Display the transformed image
            st.image(transformed_image, caption="Transformed Image.", use_column_width=True)

            # Save user details and selected image in session state
            st.session_state.user_details = {"k": k, "sigma": sigma}
            st.session_state.selected_image = transformed_image

            # Go to the next page
            st.rerun()

if __name__ == "__main__":
    if "user_details" not in st.session_state:
        main()
    else:
        # Display the image on the second page
        st.title("Waterpixels - Image Viewer")
        st.image(st.session_state.selected_image, caption="Uploaded Image.", use_column_width=True)
        st.write("Details provided by the user:")
        st.write(f"- Regularization Constant (k): {st.session_state.user_details['k']}")
        st.write(f"- Step (sigma): {st.session_state.user_details['sigma']}")
