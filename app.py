from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Streamlitページの設定
st.set_page_config(page_title="エキスパートLLMアシスタント", layout="wide")
st.title("🤖 エキスパートLLMアシスタント")
st.markdown("""
### このアプリの使い方:
1. ラジオボタンからエキスパートの種類を選択してください
2. テキストエリアに質問やプロンプトを入力してください
3. 送信ボタンをクリックしてエキスパートの回答を取得してください
""")

# エキスパートの種類とそのシステムメッセージを定義
experts = {
	"プログラマー": "あなたはプログラミング言語、デザインパターン、ベストプラクティスに関する深い知識を持つエキスパートソフトウェアエンジニアです。",
	"データサイエンティスト": "あなたは機械学習、統計分析、データ可視化に関する豊富な経験を持つエキスパートデータサイエンティストです。",
	"ライター": "あなたはさまざまなジャンルで魅力的で明確かつ説得力のあるコンテンツを作成する才能を持つエキスパートライターです。"
}

def get_expert_response(user_input: str, expert_type: str) -> str:
	"""
	指定されたエキスパートの人格でLLMからの応答を取得します。
	
	Args:
		user_input: ユーザーの質問やプロンプト
		expert_type: 選択されたエキスパートの種類
	
	Returns:
		LLMの応答
	"""
	llm = OpenAI(temperature=0.7)
	system_message = experts[expert_type]
	
	template = f"{system_message}\n\nユーザーの質問: {{input}}"
	prompt = PromptTemplate(input_variables=["input"], template=template)
		
	response = llm.invoke(prompt.format(input=user_input))
	return response

# サイドバーでエキスパートを選択
st.sidebar.header("エキスパートの種類を選択")
selected_expert = st.sidebar.radio("エキスパートを選んでください:", list(experts.keys()))

# メインの入力エリア
user_question = st.text_area("質問やプロンプトを入力してください:", height=150)

if st.button("送信"):
	if user_question.strip():
		with st.spinner("Thinking..."):
			response = get_expert_response(user_question, selected_expert)
		st.success("応答が取得されました!")
		st.write(response)
	else:
		st.warning("まず質問を入力してください。")




