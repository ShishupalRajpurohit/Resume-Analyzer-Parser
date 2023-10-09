# IMPORT LIBRARIES
import re
import PyPDF2
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st
import nltk

nltk.download('wordnet')
import en_core_web_sm

nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# ---------------------------------------------------------------------------------------------------------------
# picture code

img_url = "https://getwallpapers.com/wallpaper/full/b/3/f/1224082-widescreen-plain-hd-wallpapers-2560x1600.jpg"

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://getwallpapers.com/wallpaper/full/b/3/f/1224082-widescreen-plain-hd-wallpapers-2560x1600.jpg");
    background-size: 195%;
    background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("{img_url}");
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0, 0, 0, 0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
# -------------------------------------------------------------------------------------------------------

st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: Yellow;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')


def extract_skills(resume_text):
    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks
    tokens = [token.text for token in nlp_text if
              not token.is_stop]  # removing stop words and implementing word tokenization

    data = pd.read_csv(r"skills.csv")  # reading the csv file
    skills = list(data.columns.values)  # extract values
    skillset = []

    for token in tokens:  # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)

    for token in noun_chunks:  # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]


def getText(filename):
    fullText = ''  # Create empty string
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para
    else:
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
        for paragraph in page_content:
            fullText = fullText + paragraph
    return (fullText)


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages = pdf.pages[0]
            resume.append(pages.extract_text())
    return resume


def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)


file_type = pd.DataFrame([], columns=['Uploaded File', 'Predicted Profile', 'Skills', ])
filename = []
predicted = []
skills = []

upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned = preprocess(display(doc_file))
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        st.write("Resume in text")
        st.write(extText)
        st.write("Resume after processing")
        st.write(cleaned)
        st.write("skills from resume")
        st.write(skills)
        st.write("Prediction")
        # MAIN CODE
        import pickle as pk

        model = pk.load(open(r'model_GradientBoost.pkl', 'rb'))
        Vectorizer = pk.load(open(r'tfidf_vector.pkl', 'rb'))
        test_resumes = []
        test_resumes.append(cleaned)
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)

if len(predicted) > 0:
    file_type['Uploaded File'] = filename
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())

select = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select)

if option == 'PeopleSoft':
    st.table(file_type[file_type['Predicted Profile'] == 'PeopleSoft'])
elif option == 'SQL Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
elif option == 'React JS Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'React JS Developer'])
elif option == 'Workday':
    st.table(file_type[file_type['Predicted Profile'] == 'Workday'])