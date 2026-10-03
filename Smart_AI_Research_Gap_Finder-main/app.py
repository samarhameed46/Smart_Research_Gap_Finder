import os
import tempfile
import streamlit as st
from orchestrator import ResearchGapFinderOrchestrator

st.set_page_config(page_title='Smart Research Gap Finder', page_icon='🧠', layout='wide')
st.markdown('''<style>.block-container{padding-top:1.5rem;max-width:1400px}.hero{padding:32px;border-radius:22px;margin-bottom:22px;background:linear-gradient(135deg,#2563eb,#7c3aed);color:white}.result-box{padding:22px;border-radius:16px;border:1px solid rgba(128,128,128,.25);line-height:1.8;overflow-wrap:break-word}.stButton>button{width:100%;height:52px;border-radius:12px;font-weight:700}</style>''', unsafe_allow_html=True)

if 'orchestrator' not in st.session_state: st.session_state.orchestrator = ResearchGapFinderOrchestrator()
if 'results' not in st.session_state: st.session_state.results = None
if 'messages' not in st.session_state: st.session_state.messages = []
orch = st.session_state.orchestrator

with st.sidebar:
    st.markdown('## 🧠 Smart Research Gap Finder')
    st.success('🟢 System Ready')
    st.markdown('### AI Stack')
    st.write('**LLM:** GPT-OSS-20B\n\n**Inference:** Groq\n\n**RAG:** FAISS + MiniLM\n\n**Workflow:** Agentic Multi-Agent')
    st.divider(); st.markdown('### Workflow')
    st.write('📄 Parse PDFs → 🧩 Structure-aware chunks → 🔍 MMR + lexical retrieval → 🤖 Multi-agent analysis → 🔎 Verification → 💡 Research ideas → 📝 Proposal')

st.markdown('<div class="hero"><h1>🧠 Smart Research Gap Finder</h1><h4>Transform Research Papers into Evidence-Based Research Opportunities</h4><p>Upload papers to run Advanced RAG, Agentic RAG and a multi-agent research workflow.</p></div>', unsafe_allow_html=True)

files = st.file_uploader('Upload research papers (PDF)', type=['pdf'], accept_multiple_files=True)
col1, col2 = st.columns(2)
with col1: st.metric('Papers', len(files) if files else 0)
with col2: st.metric('RAG', 'Advanced + Agentic')

if st.button('🚀 Analyze Papers', type='primary', disabled=not files):
    with st.spinner('Running Advanced RAG and research agents...'):
        paths=[]
        try:
            for f in files:
                fd, path = tempfile.mkstemp(suffix='.pdf'); os.close(fd)
                with open(path,'wb') as out: out.write(f.getbuffer())
                paths.append(path)
            st.session_state.results = orch.process_papers(paths)
        finally:
            for p in paths:
                try: os.remove(p)
                except OSError: pass

results = st.session_state.results
if results:
    if 'error' in results: st.error(results['error'])
    else:
        st.subheader('Agent Activity')
        for item in orch.state.agent_log: st.success('✓ ' + item)
        tabs = st.tabs(['Summary','Comparison','Trends','Limitations / Review','Potential Gaps','Verified Gaps','Research Ideas','Proposal'])
        keys = ['summary','comparison','trends','limitations','research_gaps','verified_gaps','research_ideas','proposal']
        for tab,key in zip(tabs,keys):
            with tab: st.markdown(results.get(key,'No result available.'))

st.divider(); st.subheader('🤖 Research Assistant')
for m in st.session_state.messages:
    with st.chat_message(m['role']): st.markdown(m['content'])
question = st.chat_input('Ask a question about your uploaded papers...')
if question:
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'): st.markdown(question)
    answer = orch.ask_chatbot(question)
    st.session_state.messages.append({'role':'assistant','content':answer})
    with st.chat_message('assistant'): st.markdown(answer)

st.caption('Smart Research Gap Finder • Advanced RAG • Agentic RAG • Multi-Agent Research Workflow')
