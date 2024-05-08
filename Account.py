from sqlalchemy import true
import streamlit as st
import mysql.connector
from hashlib import sha256
def config():
    st.set_page_config(
        page_icon="📊",  # You can replace this with the path to your custom icon
        page_title="SCAPT"
    )

# Call the config function to set up Streamlit app configuration
config()

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

# UI layout
def main():
    

    if 'login' not in st.session_state:
        st.session_state['login'] = False
        st.session_state['username'] = ''
    
    if st.session_state['login']:
        st.subheader(f"Welcome {st.session_state['username']}!")
    
    # Profile editing section
        with st.expander("Edit Profile"):
            user_info = get_user_info(st.session_state['username'])
            if user_info:
                new_username = st.text_input("Username", user_info['username'])
                new_email = st.text_input("Email", user_info['email'])
                new_contact = st.text_input("Contact No", user_info['contact'])
        
                if st.button("Update Profile"):
                    update_user_profile(st.session_state['username'], new_username, new_email, new_contact)
                    st.session_state['username'] = new_username
                    st.success("Profile updated successfully!")
        if st.button("Logout 👋"):
            st.session_state['login'] = False
            st.session_state['username'] = ''
            st.success("You have been logged out.")
    else:
        menu = ["Login", "SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Login":
            st.subheader("Login Section")

            username = st.text_input("User Name")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                create_users_table()
                hashed_password = hash_password(password)
                result = login_user(username, hashed_password)
                if result:
                    st.session_state['login'] = True
                    st.session_state['username'] = username
                else:
                    st.error("Invalid  Login Credentials. Please Try again !!")
                    
                   

        elif choice == "SignUp":
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_email = st.text_input("Email")  # Add this line for email input
            new_password = st.text_input("Password", type='password')

            if st.button("SignUp"):
                create_users_table()
        # Add a new function to validate the email format
                if validate_email(new_email):
                    add_user(new_user, hash_password(new_password), new_email)  
                    st.markdown("<meta http-equiv='refresh' content='0;url=/Account' />", unsafe_allow_html=True)
                    st.success("You have successfully created an account!")
                    st.info("Go to the Login Menu to login")
                else:
                    st.error("Enter a valid email address")


# Database connection
def create_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="disha13",
        database="mydatabase"
    )

# Create users table

# Modify the users_table creation to include a contact column
def create_users_table():
    db = create_db_connection()
    cursor = db.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS users_table(username VARCHAR(255), password VARCHAR(255), email VARCHAR(255), contact VARCHAR(255))')
    db.close()

# Add user data to the table (modified to include email)
def add_user(username, password, email):
    db = create_db_connection()
    cursor = db.cursor()
    cursor.execute('INSERT INTO users_table(username, password, email) VALUES (%s,%s,%s)', (username, password, email))
    db.commit()
    db.close()

# Function to validate email format (simple regex example)
import re
def validate_email(email):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.fullmatch(pattern, email)

# Verify user login
def login_user(username, password):
    db = create_db_connection()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users_table WHERE username = %s AND password = %s', (username, password))
    data = cursor.fetchall()
    db.close()
    return data

# Hash passwords
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# Function to get user information
def get_user_info(username):
    db = create_db_connection()
    cursor = db.cursor()
    cursor.execute('SELECT username, email, contact FROM users_table WHERE username = %s', (username,))
    user_info = cursor.fetchone()
    if user_info:
        return {'username': user_info[0], 'email': user_info[1], 'contact': user_info[2]}
    else:
        return None
    db.close()

# Function to update user profile
def update_user_profile(old_username, new_username, new_email, new_contact):
    db = create_db_connection()
    cursor = db.cursor()
    cursor.execute('UPDATE users_table SET username = %s, email = %s, contact = %s WHERE username = %s', (new_username, new_email, new_contact, old_username))
    db.commit()
    db.close()

if __name__ == '__main__':
    main()
