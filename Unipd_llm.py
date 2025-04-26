import pandas as pd
import re
import xml.etree.ElementTree as ET
import nltk
import pyterrier as pt
import json
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from transformers import T5ForConditionalGeneration, T5Tokenizer
nltk.data.path.append('C:/Users/ASUS/Documents/BTP_2_Code_base/Clinical-Trial-Retriever/Data')
import openai
openai.api_key = "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"
# Check if the script is being run with a valid argument
if len(sys.argv) != 2:
    print("Usage: python Unipd.py <Run_1|Run_2|Run_3|Run_4>")
    sys.exit(1)

# Get the run mode from the command line argument
run_mode = sys.argv[1]

'''
Read and process an XML file containing clinical trial topics.
'''

tree = ET.parse('C:/Users/ASUS/Documents/BTP_2_Code_base/Clinical-Trial-Retriever/Information Retriever/topics.xml')
root = tree.getroot()


# Define regex patterns for age and gender
'''
Extract metadata (age,gender) from clinical trial topics.
'''
age_pattern = r'\b(\d{1,2})\s*[-]?year[-]?old\b|\b(\d{1,2})\s*[-]?month[-]?old\b|\b(\d{1,2})\s*[-]?week[-]?old\b' # regular expression for age extraction
gender_patterns = {
    'male': r'\b(male|man|boy)\b',
    'female': r'\b(female|woman|girl)\b',
    'neutral': r'\b(infant|baby)\b',
} # regex pattern for gender extraction



'''
in the loop below we basically try to process the XML data
to extract structured information (age,gender) from each clinical trial topic and store it in a list.
The extracted information is then used to create a pandas DataFrame for further analysis.
'''
data = []

for topic in root.findall('topic'): # LOOP through each topic in the XML file
    # Extract topic number and text
    topic_number = topic.get('number')
    text = topic.text.strip()
    
    # Extracting age information using regex
    age_match = re.search(age_pattern, text)
    if age_match:
        if age_match.group(1):  # Year old
            age = int(age_match.group(1))
        elif age_match.group(2):  # Month old
            age = f"{age_match.group(2)} month old"
        elif age_match.group(3):  # Week old
            age = f"{age_match.group(3)} week old"
    else:
        age = None
    
    # Extracting gender information using regex
    gender = None
    if re.search(gender_patterns['female'], text):
        gender = 'female'
    elif re.search(gender_patterns['male'], text):
        gender = 'male'
    elif re.search(gender_patterns['neutral'], text):
        gender = 'neutral'

    #  storing the extracted information in a dictionary
    # and appending it to the data list
    data.append({
        'id': topic_number,
        'text': text,
        'age': age,
        'gender': gender
    })

# Creating a DataFrame from the extracted data
df = pd.DataFrame(data)

print(df)

# Initializing NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))









# Functions for NLS, KS, and RM3 Expansion
'''
a function called split_text_by_fullstop() that processes a given text by splitting it into smaller segments (sentences or phrases) based on periods (.). 
'''
def split_text_by_fullstop(text):
    text = text.lower() # convert text to lowercase
    # text.split(".") splits the text into segments based on periods
    # [segment.strip() for segment in text.split(".") if segment.strip()] removes leading/trailing whitespace and filters out empty segments
    segments = [segment.strip() for segment in text.split('.') if segment.strip()]
    return segments # a cleaned list of segments


'''
 minimal_summary(segment), is designed to filter 
 and format medical text segments to retain only those
   containing clinically relevant keywords.
'''
'''
medical keyword filtering function

1. Checks if a text segment (from split_text_by_fullstop()) contains any clinically important terms.

2. If yes, ensures the segment ends with a . and returns it.

3. If no keywords are found, returns an empty string '' (discards the segment).
'''
def minimal_summary(segment):
    keywords = ['boy', 'girl', 'male', 'female', 'man', 'woman', 'diagnosed', 'echocardiography', 'physical',
                'ct scan', 'fluid', 'biopsy', 'testing', 'exam', 'examination', 'laboratory', 'the patient shows',
                'evaluation', 'characteristics', 'medical', 'abnormal', 'ultrasound', 'uterus', 'testes', 'X-ray',
                'endoscopy', 'chronic', 'characterized', 'observed', 'auscultation', 'history', 'mri', 'hemophilia',
                'lipoma', 'otoscopy', 'diagnosis', 'diagnosed', 'mslt']
    
    words = set(segment.lower().split())
    if any(keyword.lower() in words for keyword in keywords):
        if not segment.endswith('.'):
            segment += '.'
        return segment
    return ''


'''
This NLS(df, run_keyword) function implements Natural Language Selection (NLS) 
to preprocess clinical trial topics by extracting medically relevant phrases.
'''
def NLS(df, run_keyword):
    if run_keyword in ["Run_1", "Run_3"]:
        nls_summaries = []
        for text in df['text']:
            segments = split_text_by_fullstop(text)
            summaries = [minimal_summary(segment) for segment in segments if minimal_summary(segment)]
            nls_summary = ' '.join(summaries) # combine valid segments
            nls_summaries.append(nls_summary) # Adds the condensed summary to the results list.
        df['nls_summary'] = nls_summaries  
    return df


'''
Extracts medically relevant keywords from text while removing noise.
'''
def extract_unique_keywords(text):
    words = word_tokenize(text.lower()) # txt is converted to lowercase and tokenized into words
    stop_words = set(stopwords.words('english')) # stop words are removed
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words] # keeps only alphabetic words
    pos_tags = pos_tag(filtered_words) # part-of-speech tagging is applied to the filtered words
    # Extract nouns and adjectives
    important_terms = [word for word, tag in pos_tags if tag in ('NN', 'JJ')]

    # Remove duplicates while preserving order
    # seen is a set to track unique terms
    seen = set()
    unique_terms = []
    for term in important_terms:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)

    return unique_terms



'''
Creates a clean search query from keywords.
'''

def build_unique_query(text):
    unique_keywords = extract_unique_keywords(text)
    query = ' '.join(unique_keywords) # joins the unique keywords into a single string
    return query

# Loading and Initializing the T5 model for text summarization
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


'''
AI-Powered Text Summarization

'''
def t5_summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

'''
text = "A 5-year-old male patient presented with high fever..."
summary = t5_summarize(text)
# Output: "young male patient with fever"
'''
def clean_query_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text) # Remove special characters and keep only alphanumeric characters and spaces

'''
WORKFLOW:
1. Preprocess the clinical trial topics using NLS.
2. Initialize searchengine (pyterrier + BM25).
3. For each topic, retrieve the top 1000 documents using BM25.
4. Save Results (TREC-style + JSON)
'''

def load_doc_text(docno):
    path = fr"C:\Users\ASUS\Downloads\Design_Lab\Clinical-Trial-Retriever\extracted txt\{docno}.txt"
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading {docno}: {e}")
        return ""


from openai import OpenAI

client = OpenAI()  # This uses your environment variable OPENAI_API_KEY

def call_llm_score(query, doc_text):
    prompt = f"""You are an expert in biomedical information retrieval.
Query: {query}
Document: {doc_text}

On a scale of 0 to 10, how relevant is this document to the query? Just reply with a number:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        score_str = response.choices[0].message.content.strip()
        return float(score_str)
    except Exception as e:
        print(f"OpenAI error: {e}")
        return 0.0





if run_mode == "Run_1":
    df = NLS(df, "Run_1") # Apply NLS to the DataFrame
    df.to_csv('output_run_1.csv', index=False)
    df['nls_summary'] = df['nls_summary'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))

    # START PYTERRIER
    if not pt.started():
       pt.init()
    index_path = 'C:/Users/ASUS/Documents/BTP_2_Code_base/Index'

    # Load the pre-built clinical trial index
    try:
        index = pt.IndexFactory.of(index_path)
        print(f"Index loaded successfully from: {index_path}")
    except Exception as e:
        print(f"Error loading index from {index_path}: {e}")
        raise

# Initialize the BM25 retriever
    retriever = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.75, "k1": 1.2})
    print("BM25 retriever initialized successfully.")

    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['nls_summary'] # get preprocessed query
        queries_df = pd.DataFrame({
            'qid': [idx],
            'query': [query]
        })

        try:
            initial_results = retriever.transform(queries_df) # Run BM25 search
            top_results = initial_results.head(50) # keep top 50 results
            
            # formatting results to get final o/p
            for rank, (_, result_row) in enumerate(top_results.iterrows(), start=1):
                docno = result_row['docno']
                docno = docno.replace('.txt', '')  # This line is indented with 4 spaces
                text_line = f"{query_id} Q0 {docno} {rank} {result_row['score']} Run_1"
                text_lines.append(text_line)
                results_list.append({
                    'topic_no': idx,
                    'docno': result_row['docno'],
                    'rank': rank,
                    'score': result_row['score']
                })
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1
        break

    text_lines = []
    results_list = []
    query_id = 1
    for idx, row in df.iterrows():
        query = row['nls_summary']
        queries_df = pd.DataFrame({
            'qid': [idx],
            'query': [query]
        })
        try:
            initial_results = retriever.transform(queries_df)
            top_results = initial_results.head(50)
            llm_ranked = []
            for _,result_row in top_results.iterrows():
                docno = result_row['docno']
                docno = docno.replace('.txt', '')
                doc_text = load_doc_text(docno)
                
                if doc_text.strip() == "":
                    print(f"Document {docno} is empty or not found.")
                    continue
                score = call_llm_score(query, doc_text)
                llm_ranked.append((docno, score))

                top_llm = sorted(llm_ranked, key=lambda x: x[1], reverse=True)[:10]
                for rank, item in enumerate (top_llm, start=1):
                    text_line = f"{query_id} Q0 {item[0]} {rank} {item[1]} Run_1"
                    text_lines.append(text_line)
                    results_list.append({
                        'topic_no': idx,
                        'docno': item[0],
                        'rank': rank,
                        'score': item[1]
                    })
        except Exception as e:
            print(f"Error during retrieval for query ID {idx}: {e}")
        query_id += 1
        break

    with open(f"Run_1_NLS_RETRIEVAL_unipd_llm.txt", 'w') as txt_file:
        txt_file.write("\n".join(text_lines) + "\n")

    with open(f"Run_1_NLS_RETRIEVAL_unipd_llm.json", 'w') as json_file:
        json.dump(results_list, json_file)
    print("Run 1 with LLM reranking completed successfully.")