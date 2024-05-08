import streamlit as st


def config():
    st.set_page_config(
        page_icon="📊",  # You can replace this with the path to your custom icon
        page_title="SCAPT"
    )

# Call the config function to set up Streamlit app configuration
config()

text_color = "white"

# Homepage content
st.markdown(f"<h1 style='text-align: center; color: {text_color};'>Welcome to SCAPT!!!</h1>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align: center; color: {text_color};'>Ensure your data remains your most valuable asset.</h1>", unsafe_allow_html=True)

# Description Section
st.markdown(
    f"""<h1 style='text-align: center; font-size: 20px; color: {text_color};'>
    SCAPT is the most versatile and secure data management platform for cleaning and maintaining CRM data in less time, so you always have report-ready data improving the effectiveness of your revenue operations.
    </h1>""",
    unsafe_allow_html=True
)

st.markdown(f"<h1 style='text-align: center; color: {text_color};'>Get Started !!</h1>", unsafe_allow_html=True)
# Set background
st.markdown(
    """
    <style>
    .stApp {
      background-image: url("https://img.freepik.com/free-photo/glowing-spaceship-orbits-planet-starry-galaxy-generated-by-ai_188544-9655.jpg?size=626&ext=jpg&ga=GA1.1.1700460183.1708560000&semt=sph");
      background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)
center_btn = st.container()

with center_btn:
   st.markdown("""
   <style>
   .stButton {
     display: flex;
     justify-content: center;
   }
   </style>
   """, unsafe_allow_html=True)




