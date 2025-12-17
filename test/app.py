from flask import Flask, request, render_template, jsonify
from transformers import pipeline
import PyPDF2
import os

# Flask 애플리케이션 설정
app = Flask(__name__)

# 텍스트 요약 파이프라인 설정
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# 질문 답변 파이프라인 설정
question_answerer = pipeline("question-answering", model="deepset/roberta-base-squad2")

# PDF에서 텍스트 추출
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# 텍스트 요약
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# 질문에 답하기
def answer_question(text, question):
    answer = question_answerer(question=question, context=text)
    return answer['answer']

# 홈 페이지 (파일 업로드 및 요약 보기)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # PDF에서 텍스트 추출
            text = extract_text_from_pdf(file_path)

            # 텍스트 요약
            summary = summarize_text(text)

            return render_template('index.html', summary=summary)

    return render_template('index.html', summary=None)

# 질문을 받는 페이지
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    text = request.form['text']

    # 질문에 대한 답변
    answer = answer_question(text, question)

    return jsonify(answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
