AI Property Search Chatbot (deployment link : https://property-chatbot-110204.streamlit.app/)
This is a proof-of-concept project I built to explore how natural language processing can make 
searching for real estate more intuitive. Instead of using a dozen dropdowns and filters, you can 
just type what you're looking for in plain English. 
The app is built entirely in Python using the Streamlit framework. 
My Approach 
I wanted to build a system that was both smart and fast. My solution uses a hybrid approach: 
1. Rule-Based Filtering: I'm using regular expressions (regex) to quickly pull out hard facts 
from the user's query, like the number of bedrooms (BHK), the maximum price, and 
location names. This is super efficient for narrowing down the search space. 
2. Semantic Search: This is where the "AI" part comes in. After filtering, I use a 
Sentence-Transformer model to rank the remaining properties. The model understands 
the meaning behind the words, not just the keywords. This means a query like 
"family-friendly homes near a park" could potentially rank properties with amenities like 
"landscaped gardens" or "play area" higher, even if the user didn't explicitly type those 
words. 
How to Run It 
First, make sure you have Python 3.8+ installed. 
1. Set up a virtual environment. It's just good practice. 
python -m venv venv 
source venv/bin/activate  # On Linux/macOS 
# venv\Scripts\activate   # On Windows 
2. Install the dependencies. 
pip install -r requirements.txt 
3. Make sure your data is in the right place. You need a data/ folder in the same directory 
as the app.py script, containing all the required CSV files. 
4. Launch the app! 
streamlit run app.py 
It should open up in your browser automatically. The first time you run it, it will take a 
minute or two to download the language model. 
Known Limitations & Future Ideas 
● Location Parsing is Basic: The current regex for location is simple. A better approach 
would be to use a proper Named Entity Recognition (NER) model or at least have a 
predefined list of cities and localities to match against. 
● No Memory: The chatbot doesn't remember previous turns in the conversation. Adding 
conversational memory would be a great next step (e.g., "now show me some in a 
cheaper area"). 
● Scalability: For a massive dataset, generating and holding all embeddings in memory 
might not be feasible. A real-world application would use a dedicated vector database like 
Pinecone or Weaviate. 
Thanks for checking out my project!