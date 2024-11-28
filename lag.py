import streamlit as st
import google.generativeai as genai
from datetime import datetime
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import hashlib
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import random
from datetime import timedelta
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
print(os.getenv("GOOGLE_API_KEY"))



def create_vector_embeddings(pdf_folder_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader(pdf_folder_path)  # Data ingestion
    docs = loader.load()  # Document loading

    # Extract PDF names
    pdf_names = [os.path.basename(doc.metadata['source']) for doc in docs]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
    final_documents = text_splitter.split_documents(docs)  # Splitting all documents
    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings

    return vectors, pdf_names

# Function for the Document Retrieval tab
def document_retrieval_tab():
    st.title("📄 Document Retrieval")

    # File path for PDFs
    pdf_folder_path = r"/workspaces/Smartlearning/us_census"

    if not os.path.exists(pdf_folder_path):
        st.error("PDF folder not found. Please ensure the path is correct.")
    else:
        pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("No PDFs found in the folder.")
        else:
            st.write("Available PDFs in the folder:")
            for pdf_file in pdf_files:
                st.write(f"- {pdf_file}")

            if st.button("🔄 Create Vector Store"):
                with st.spinner("Creating vector store..."):
                    vectors, pdf_names = create_vector_embeddings(pdf_folder_path)
                    st.success("Vector Store DB is ready!")
                    
                    # Session storage for vectors and document names
                    st.session_state["vectors"] = vectors
                    st.session_state["pdf_names"] = pdf_names

            if "vectors" in st.session_state:
                query = st.text_input("Enter your query:")
                if query:
                    with st.spinner("Retrieving answer..."):
                        try:
                            # Initialize LLM and chain
                            llm = ChatGroq(
                                groq_api_key=os.getenv('GROQ_API_KEY'),
                                model_name="Llama3-8b-8192"
                            )
                            prompt = ChatPromptTemplate.from_template(
                                """
                                Answer the questions based on the provided context only.
                                Please provide the most accurate response based on the question.
                                <context>
                                {context}
                                <context>
                                Questions: {input}
                                """
                            )
                            document_chain = create_stuff_documents_chain(llm, prompt)
                            retriever = st.session_state["vectors"].as_retriever()
                            retrieval_chain = create_retrieval_chain(retriever, document_chain)

                            # Query the chain
                            start_time = time.process_time()
                            response = retrieval_chain.invoke({'input': query})
                            response_time = time.process_time() - start_time

                            st.write(f"Response time: {response_time:.2f} seconds")
                            st.subheader("Answer:")
                            st.write(response['answer'])

                            st.subheader("Relevant Documents:")
                            for i, doc in enumerate(response["context"]):
                                with st.expander(f"Document {i+1}"):
                                    st.write(doc.page_content)
                        except Exception as e:
                            st.error(f"Error retrieving documents: {str(e)}")

# Database setup
def init_db():
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            user_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_user_data(username, data):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('UPDATE users SET user_data = ? WHERE username = ?', 
              (json.dumps(data), username))
    conn.commit()
    conn.close()

def load_user_data(username):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('SELECT user_data FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0]:
        return json.loads(result[0])
    return None

# Initialize database
init_db()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Set up Google Gemini API configuration
genai.configure(api_key=GEMINI_API_KEY)

# Model Configuration
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Page Configuration
st.set_page_config(
    page_title="Smart Learning Hub",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication functions
def create_user(username, password):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password, user_data) VALUES (?, ?, ?)',
                 (username, hash_password(password), json.dumps({
                     'learning_paths': {},
                     'study_time': {},
                     'achievements': [],
                     'goals': [],
                     'learning_style': None
                 })))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('learning_hub.db')
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

# Login/Register UI
def show_login_page():
    st.title("🎓 Smart Learning Hub")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if verify_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state.user_data = load_user_data(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Register")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                elif create_user(new_username, new_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists")

# Modified Learning Style Assessment Page
def show_learning_style_assessment():
    st.title("🧠 Discover Your Learning Style")
    
    learning_style_questions = {
        "scenario_1": {
            "question": "When learning something new, which method helps you remember best?",
            "options": [
                "Seeing diagrams and pictures",
                "Listening to explanations",
                "Hands-on practice",
                "Reading and taking notes"
            ]
        },
        "scenario_2": {
            "question": "When following instructions, you prefer to:",
            "options": [
                "Look at diagrams or demonstrations",
                "Hear verbal explanations",
                "Try it out yourself",
                "Read written instructions"
            ]
        },
        "scenario_3": {
            "question": "When studying, you tend to:",
            "options": [
                "Use charts and visual aids",
                "Discuss topics with others",
                "Create models or practice examples",
                "Make written summaries"
            ]
        }
    }
    
    with st.form("learning_style_form"):
        responses = {}
        for key, data in learning_style_questions.items():
            responses[key] = st.radio(
                data["question"],
                data["options"],
                key=f"style_{key}"
            )
        
        submit = st.form_submit_button("Analyze My Learning Style")
        
        if submit:
            with st.spinner("Analyzing your learning style..."):
                analysis = analyze_learning_style(responses)
                st.session_state.user_data['learning_style'] = analysis
                
                st.success(f"Your primary learning style: {analysis['style']}")
                
                tab1, tab2, tab3 = st.tabs(["Understanding", "Strategies", "Recommended Videos"])
                
                with tab1:
                    st.write("### Understanding Your Style")
                    st.write(analysis['explanation'])
                
                with tab2:
                    st.write("### Recommended Strategies")
                    for strategy in analysis['strategies']:
                        st.write(f"• {strategy}")
                    
                    st.write("### Recommended Tools")
                    for tool in analysis['tools']:
                        st.write(f"• {tool}")
                
                with tab3:
                    st.write("### Recommended Videos")
                    if analysis.get('recommended_videos'):
                        for video in analysis['recommended_videos']:
                            with st.expander(video['title']):
                                st.write(video['description'])
                                st.markdown(f"[Watch Video]({video['url']})")
                    else:
                        st.info("No specific video recommendations available.")

# Modified Resources Page
# def show_resources_page():
#     st.title("📚 Learning Resources")
    
#     if not st.session_state.user_data['learning_paths']:
#         st.info("Please create a learning path first to get personalized resources!")
#     else:
#         selected_path = st.selectbox(
#             "Select Learning Path:",
#             list(st.session_state.user_data['learning_paths'].keys())
#         )
        
#         if selected_path:
#             learning_style = st.session_state.user_data.get('learning_style', {}).get('style', 'General')
#             resources_content = generate_resources(selected_path, learning_style)
            
#             categories = [
#                 "ONLINE COURSES",
#                 "BOOKS AND MATERIALS",
#                 "PRACTICE PLATFORMS",
#                 "COMMUNITY RESOURCES",
#                 "VIDEO TUTORIALS"
#             ]
            
#             for i, category in enumerate(categories):
#                 with st.expander(f"📌 {category}"):
#                     try:
#                         sections = resources_content.split(category + ":")
#                         if len(sections) > 1:
#                             content = sections[1]
#                             if i < len(categories) - 1:
#                                 content = content.split(categories[i + 1] + ":")[0]
#                             st.markdown(content.strip())
#                         else:
#                             st.write("No resources available for this category.")
#                     except Exception as e:
#                         st.write("Error displaying resources for this category.")
#                         continue

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    show_login_page()
else:
    # Learning Style Assessment Function
    def analyze_learning_style(responses):
        prompt = f"""
        Based on the following learning style assessment responses:
        {json.dumps(responses, indent=2)}
        
        Please analyze the learning style and provide a response in the following format:
        {{
            "style": "Visual/Auditory/Kinesthetic/Reading-Writing",
            "explanation": "Detailed explanation of the learning style",
            "strategies": ["Strategy 1", "Strategy 2", "Strategy 3"],
            "tools": ["Tool/Resource 1", "Tool/Resource 2", "Tool/Resource 3"],
            "recommended_videos": [
            {{
                "title": "Video Title 1",
                "url": "https://www.youtube.com/watch?v=...",
                "description": "Brief description of the video content"
            }},
            {{
                "title": "Video Title 2",
                "url": "https://www.youtube.com/watch?v=...",
                "description": "Brief description of the video content"
            }}
        }}
        Ensure the video recommendations are real, popular educational videos relevant to the learning style.
        """
        
        try:
            response = model.generate_content(prompt)
            # Ensure the response is properly formatted as JSON
            result = response.text.strip()
            if not result.startswith('{'):
                result = result[result.find('{'):result.rfind('}')+1]
            return json.loads(result)
        except Exception as e:
            st.error(f"Error analyzing learning style. Please try again. Error: {str(e)}")
            return {
                "style": "General",
                "explanation": "Unable to determine specific learning style. Using general learning approach.",
                "strategies": ["Mixed learning approach", "Multiple resource types", "Regular practice"],
                "tools": ["Online courses", "Educational videos", "Practice exercises"],
                "recommended_videos": []
            }

    # Resource Generation Function
    def generate_resources(path, learning_style="General"):
        prompt = f"""
        As an expert educator, suggest detailed, high-quality learning resources for {path} tailored for {learning_style} learners.
        Focus only on widely available, reputable sources.
        
        Structure your response exactly like this:

        ONLINE COURSES:
        - Provide 3-4 popular courses from Coursera, edX, Udemy with:
            - Exact course names that exist on these platforms
            - Brief description focusing on key benefits
            - Approximate duration and difficulty level
            - Main topics covered
        
        BOOKS AND MATERIALS:
        - List 3-4 widely recognized books with:
            - Full title and author name
            - Publisher and year if recent
            - Key topics and target audience
            - Why it's particularly good for {learning_style} learners
        
        PRACTICE PLATFORMS:
        - Suggest 2-3 interactive platforms with:
            - Platform name and main focus
            - Types of exercises/projects available
            - Skill level requirements
            - Any free vs paid considerations
        
        LEARNING COMMUNITIES:
        - Recommend 2-3 active communities with:
            - Community name and platform (Reddit, Discord, Stack Exchange, etc.)
            - Main focus and activity type
            - Why it's valuable for learners
            - Any entry requirements or guidelines
        
        VIDEO RESOURCES:
        - Suggest 2-3 high-quality video channels or series with:
            - Channel/Creator name
            - Content style and format
            - Best playlists or series to start with
            - Why it works well for {learning_style} learning style

        Focus on currently active and maintained resources. For each resource, explain why it's particularly suitable for {learning_style} learners.
        Do not include specific URLs, as these can change. Instead, provide enough information for users to easily find the resources.
        """
        
        try:
            response = model.generate_content(prompt)
            resources_content = response.text.strip()
            return resources_content
        except Exception as e:
            return f"Error generating resources: {str(e)}"

    def show_resources_page():
        st.title("📚 Learning Resources")
        
        if not st.session_state.user_data['learning_paths']:
            st.info("Please create a learning path first to get personalized resources!")
            return

        # Get user's learning style
        learning_style = st.session_state.user_data.get('learning_style', {}).get('style', 'General')
        
        # Path selection
        selected_path = st.selectbox(
            "Select Learning Path:",
            list(st.session_state.user_data['learning_paths'].keys())
        )
        
        if selected_path:
            with st.spinner("Generating personalized resources..."):
                resources_content = generate_resources(selected_path, learning_style)
                
                # Display resources by category
                categories = [
                    "ONLINE COURSES",
                    "BOOKS AND MATERIALS",
                    "PRACTICE PLATFORMS",
                    "LEARNING COMMUNITIES",
                    "VIDEO RESOURCES"
                ]
                
                for category in categories:
                    with st.expander(f"📌 {category}", expanded=True):
                        try:
                            # Extract category content
                            if category in resources_content:
                                section = resources_content.split(category + ":")[1]
                                # Get content until next category or end
                                end_pos = len(section)
                                for next_cat in categories:
                                    if next_cat in section:
                                        end_pos = min(end_pos, section.index(next_cat))
                                content = section[:end_pos].strip()
                                
                                # Display content with improved formatting
                                for line in content.split('\n'):
                                    if line.strip():
                                        if line.startswith('-'):
                                            st.markdown(f"**{line.strip('-').strip()}**")
                                        else:
                                            st.write(line.strip())
                        except Exception as e:
                            st.write("No resources available for this category.")
            # Add refresh button
            if st.button("🔄 Refresh Resources"):
                                       st.write("No resources available for this category.")
        
        # # Add refresh button
        # if st.button("🔄 Refresh Resources"):
        #     st.experimental_rerun()
        
        # Add feedback section
        st.markdown("---")
        st.subheader("📝 Resource Feedback")
        feedback = st.text_area("Help us improve! Let us know if any resources are outdated or particularly helpful:")
        if st.button("Submit Feedback"):
            # Store feedback in user data
            if 'resource_feedback' not in st.session_state.user_data:
                st.session_state.user_data['resource_feedback'] = []
            st.session_state.user_data['resource_feedback'].append({
                'path': selected_path,
                'feedback': feedback,
                'date': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("Thank you for your feedback!")
    
    # Dashboard Components
    def create_study_streak_chart():
        dates = pd.date_range(end=datetime.now(), periods=30)
        activity = [random.choice([0, 1, 1, 1]) for _ in range(30)]  # Simulate activity
        df = pd.DataFrame({'date': dates, 'activity': activity})
        
        fig = px.bar(df, x='date', y='activity', 
                    title='Study Streak - Last 30 Days',
                    labels={'activity': 'Study Sessions', 'date': 'Date'})
        return fig

    def calculate_learning_metrics():
        total_paths = len(st.session_state.user_data['learning_paths'])
        completed_items = sum(len(path['completed']) for path in st.session_state.user_data['learning_paths'].values())
        total_items = sum(len(path['checklist']) for path in st.session_state.user_data['learning_paths'].values())
        completion_rate = (completed_items / total_items * 100) if total_items > 0 else 0
        
        return {
            'total_paths': total_paths,
            'completion_rate': completion_rate,
            'active_days': random.randint(1, 30),  # Simulated data
            'study_hours': random.randint(10, 100)  # Simulated data
        }
    
    def get_learning_recommendations():
        if not st.session_state.user_data.get('learning_style'):
            return ["Complete your learning style assessment to get personalized recommendations!"]
        
        style = st.session_state.user_data['learning_style']['style']
        return [
            f"Based on your {style} learning style:",
            "• " + random.choice([
                "Try creating mind maps for complex topics",
                "Record yourself explaining concepts",
                "Use interactive coding environments",
                "Write detailed study notes"
            ]),
            "• " + random.choice([
                "Take regular breaks every 25 minutes",
                "Join study groups for discussion",
                "Practice with real-world projects",
                "Create flashcards for key concepts"
            ]),
            "• " + random.choice([
                "Watch video tutorials with closed captions",
                "Explain concepts to others",
                "Use physical objects to model problems",
                "Summarize readings in your own words"
            ])
        ]
    # Main Dashboard Page
    def show_dashboard():
        st.title("🎓 Learning Dashboard")
        
        # Top metrics
        metrics = calculate_learning_metrics()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Learning Paths", metrics['total_paths'])
        with col2:
            st.metric("Completion Rate", f"{metrics['completion_rate']:.1f}%")
        with col3:
            st.metric("Active Days", metrics['active_days'])
        with col4:
            st.metric("Study Hours", metrics['study_hours'])
        
        # Study streak chart
        # st.plotly_chart(create_study_streak_chart(), use_container_width=True)
        
        # Recent activity and recommendations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Recent Activity")
            if st.session_state.user_data['learning_paths']:
                for path, data in st.session_state.user_data['learning_paths'].items():
                    with st.expander(f"📚 {path}"):
                        progress = len(data['completed']) / len(data['checklist']) * 100
                        st.progress(progress / 100)
                        st.write(f"Progress: {progress:.1f}%")
                        st.write("Recent completions:")
                        for item in data['completed'][-3:]:
                            st.write(f"✅ {item}")
            else:
                st.info("Start a learning path to track your progress!")
        
        with col2:
            st.subheader("💡 Recommendations")
            recommendations = get_learning_recommendations()
            for rec in recommendations:
                st.write(rec)
            
            if st.button("Get New Tips"):
                st.rerun()

    # Sidebar Navigation
    with st.sidebar:
        st.image("https://media.licdn.com/dms/image/v2/D4D12AQHGJ3I-fJELnA/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1715845287543?e=2147483647&v=beta&t=D6L7ZHUy8qh3aFQsyeAGv0wgXB5UskmGNIwe2OETaig", caption="Smart Learning Hub")
        st.write(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout"):
            # Save user data before logout
            save_user_data(st.session_state['username'], st.session_state.user_data)
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state.user_data = None
            st.rerun()
            
        page = st.radio(
            "Navigation",
            ["Dashboard", "Learning Style Assessment", "Learning Path", "Progress Tracking", "Resources", "Document Retrieval"]
        )

# Resources Page with Error Handling
    if page == "Resources":
        show_resources_page()
    elif  page == "Document Retrieval":
        document_retrieval_tab()


    # Learning Style Assessment Page
    elif page == "Learning Style Assessment":
        st.title("🧠 Discover Your Learning Style")
        
        learning_style_questions = {
            "scenario_1": {
                "question": "When learning something new, which method helps you remember best?",
                "options": [
                    "Seeing diagrams and pictures",
                    "Listening to explanations",
                    "Hands-on practice",
                    "Reading and taking notes"
                ]
            },
            "scenario_2": {
                "question": "When following instructions, you prefer to:",
                "options": [
                    "Look at diagrams or demonstrations",
                    "Hear verbal explanations",
                    "Try it out yourself",
                    "Read written instructions"
                ]
            },
            "scenario_3": {
                "question": "When studying, you tend to:",
                "options": [
                    "Use charts and visual aids",
                    "Discuss topics with others",
                    "Create models or practice examples",
                    "Make written summaries"
                ]
            }
        }
        
        with st.form("learning_style_form"):
            responses = {}
            for key, data in learning_style_questions.items():
                responses[key] = st.radio(
                    data["question"],
                    data["options"],
                    key=f"style_{key}"
                )
            
            submit = st.form_submit_button("Analyze My Learning Style")
            
            if submit:
                with st.spinner("Analyzing your learning style..."):
                    analysis = analyze_learning_style(responses)
                    st.session_state.user_data['learning_style'] = analysis
                    
                    st.success(f"Your primary learning style: {analysis['style']}")
                    st.write("### Understanding Your Style")
                    st.write(analysis['explanation'])
                    
                    st.write("### Recommended Strategies")
                    for strategy in analysis['strategies']:
                        st.write(f"• {strategy}")
                    
                    st.write("### Recommended Tools")
                    for tool in analysis['tools']:
                        st.write(f"• {tool}")

    # Learning Path Page
    elif page == "Learning Path":
        st.title("🎯 Create Your Learning Path")
        
        learning_style = st.session_state.user_data.get('learning_style')
        if not learning_style:
            st.warning("Complete the Learning Style Assessment first for a personalized learning path!")
            if st.button("Go to Learning Style Assessment"):
                st.switch_page("Learning Style Assessment")
        else:
            st.success(f"Creating path optimized for your {learning_style['style']} learning style")
            
            goal_type = st.radio("Select your goal type:", 
                                ["New Career Path", "Skill Enhancement", "Certification"])
            
            target = st.text_input("Enter your target role or skill:")
            
            if target and st.button("Generate Learning Path"):
                with st.spinner("Generating your personalized learning path..."):
                    prompt = f"""
                    Create a detailed learning path for {target} optimized for {learning_style['style']} learners.
                    Include:
                    1. Prerequisites
                    2. Core Topics
                    3. Practical Projects
                    4. Assessment Methods
                    
                    Format as a clear, bulleted list.
                    """
                    
                    try:
                        response = model.generate_content(prompt)
                        path_content = response.text.strip()
                        
                        st.session_state.user_data['learning_paths'][target] = {
                            'checklist': [item.strip() for item in path_content.split('\n') if item.strip()],
                            'completed': [],
                            'start_date': datetime.now().strftime("%Y-%m-%d"),
                            'last_updated': datetime.now().strftime("%Y-%m-%d")
                        }
                        
                        st.success("Learning path generated successfully!")
                        st.markdown(path_content)
                    except Exception as e:
                        st.error(f"Error generating learning path: {str(e)}")


    # Progress Tracking Page
    elif page == "Progress Tracking":
        st.title("📈 Progress Tracking")
        
        for path, data in st.session_state.user_data['learning_paths'].items():
            st.subheader(f"Path: {path}")
            
            # Progress calculation
            total_items = len(data['checklist'])
            completed_items = len(data['completed'])
            progress = (completed_items / total_items * 100) if total_items > 0 else 0
            
            # Display progress
            st.progress(progress / 100)
            st.write(f"Progress: {progress:.1f}%")
            
            # Checklist
            for item in data['checklist']:
                checked = item in data['completed']
                if st.checkbox(item, value=checked, key=f"check_{path}_{item}"):
                    if item not in data['completed']:
                        data['completed'].append(item)
                elif item in data['completed']:
                    data['completed'].remove(item)

    

    elif page == "Dashboard":
        show_dashboard()

    # Auto-save user data periodically
    if st.session_state.get('username'):
        save_user_data(st.session_state['username'], st.session_state.user_data)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Smart Learning Hub - Your Personalized Learning Assistant</p>
        <p style='font-size: 0.8em'>Version 2.2</p>
    </div>
    """,
    unsafe_allow_html=True
)
