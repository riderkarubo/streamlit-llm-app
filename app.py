from dotenv import load_dotenv
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 環境変数の読み込み
load_dotenv()

# ページ設定
st.set_page_config(
    page_title="専門家AIコンサルタント",
    page_icon="🤖",
    layout="wide"
)

# 専門家の定義（トークン数を削減するため簡潔化）
# 
# 使用モデル: gpt-4o-mini（コスト効率の良いモデル）
# - 入力: $0.15 / 100万トークン
# - 出力: $0.60 / 100万トークン
#
EXPERTS = {
    "A": {
        "name": "データサイエンティスト",
        "system_message": "データサイエンティスト。データ分析・機械学習・統計に精通。データと統計的根拠を示して説明。"
    },
    "B": {
        "name": "ソフトウェアエンジニア",
        "system_message": "ソフトウェアエンジニア。プログラミング・システム設計・アーキテクチャに精通。コード例と実装方法を示して説明。"
    },
    "C": {
        "name": "マーケティングスペシャリスト",
        "system_message": "マーケティングスペシャリスト。ブランディング・デジタルマーケティング・戦略に精通。市場動向を考慮した実践的な施策を提案。"
    },
    "D": {
        "name": "財務アナリスト",
        "system_message": "財務アナリスト。財務分析・投資判断・予算管理に精通。数値・財務指標を使い、リスクとリターンを考慮して説明。"
    }
}


def get_llm_response(user_input: str, expert_choice: str) -> str:
    """
    入力テキストと専門家選択を引数として受け取り、LLMからの回答を返す関数
    
    Args:
        user_input: ユーザーが入力したテキスト
        expert_choice: 選択された専門家（A, B, C, Dのいずれか）
    
    Returns:
        LLMからの回答テキスト
    """
    # APIキーの確認
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "エラー: OPENAI_API_KEYが環境変数に設定されていません。.envファイルにOPENAI_API_KEYを設定してください。"
    
    try:
        # 選択された専門家の情報を取得
        expert = EXPERTS.get(expert_choice)
        if not expert:
            return "エラー: 無効な専門家が選択されました。"
        
        # ChatOpenAIのインスタンスを作成
        # gpt-4o-mini: コスト効率が良いモデル（入力$0.15、出力$0.60/100万トークン）
        # temperature=0.6: 予測可能性を高め、再試行を減らしてコスト削減
        # max_tokens=1500: 過剰に長い回答を抑制し、コストを削減（必要に応じて調整可能）
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            max_tokens=1500,
            api_key=api_key
        )
        
        # プロンプトテンプレートを作成
        messages = [
            SystemMessage(content=expert["system_message"]),
            HumanMessage(content=user_input)
        ]
        
        # LLMにプロンプトを渡して回答を取得
        response = llm.invoke(messages)
        
        return response.content
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"


def main():
    # タイトルと説明
    st.title("🤖 専門家AIコンサルタント")
    st.markdown("---")
    
    st.markdown("""
    ### 📖 アプリの概要
    このアプリは、4人の専門家（データサイエンティスト、ソフトウェアエンジニア、マーケティングスペシャリスト、財務アナリスト）の中から選択して、
    LLMにその専門家として振る舞わせることで、専門的なアドバイスや回答を得ることができます。
    
    ### 🎯 操作方法
    1. **専門家を選択**: ラジオボタンから、質問したい領域の専門家を選択してください
    2. **質問を入力**: テキスト入力フォームに質問や相談内容を入力してください
    3. **送信**: 「回答を取得」ボタンをクリックして、専門家からの回答を確認してください
    
    ---
    """)
    
    # サイドバーに専門家の説明を表示
    with st.sidebar:
        st.header("👥 専門家一覧")
        for key, expert in EXPERTS.items():
            st.markdown(f"**{key}. {expert['name']}**")
            st.caption(f"{expert['system_message'][:50]}...")
            st.markdown("---")
    
    # 専門家選択のラジオボタン
    st.subheader("専門家を選択してください")
    expert_choice = st.radio(
        "専門家",
        options=list(EXPERTS.keys()),
        format_func=lambda x: f"{x}: {EXPERTS[x]['name']}",
        horizontal=True,
        index=0
    )
    
    st.markdown("---")
    
    # 入力フォーム
    st.subheader("質問を入力してください")
    user_input = st.text_area(
        "入力テキスト",
        placeholder="例: Pythonでデータ分析を行う際のベストプラクティスを教えてください",
        height=150
    )
    
    # 送信ボタン
    if st.button("回答を取得", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ 質問を入力してください。")
        else:
            # ローディング表示
            with st.spinner("専門家が回答を考えています..."):
                # LLMからの回答を取得
                response = get_llm_response(user_input, expert_choice)
            
            # 回答を表示
            st.markdown("---")
            st.subheader(f"💬 {EXPERTS[expert_choice]['name']}からの回答")
            st.markdown(response)
    
    # フッター
    st.markdown("---")
    st.caption("💡 このアプリはLangChainとOpenAI GPTを使用しています。")


if __name__ == "__main__":
    main()
