import hashlib
import tempfile
from pathlib import Path
import streamlit as st
from core import Ollama, pdf_records, build_index, retrieve, answer_pdf

st.set_page_config(page_title='Multimodal AI Chat',page_icon='💬',layout='centered')
st.title('Multimodal AI Chat')
st.caption('Local models • text, images, audio and PDF retrieval')
model=Ollama()
mode=st.sidebar.radio('Mode',['Text','Image','Audio','PDF question answering'])
@st.cache_resource
def whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel('base',device='cpu',compute_type='int8')
try:
    if mode=='Text':
        prompt=st.text_area('Ask a question')
        if st.button('Generate') and prompt.strip(): st.write(model.chat('mistral',prompt))
    elif mode=='Image':
        upload=st.file_uploader('Image',type=['png','jpg','jpeg']); prompt=st.text_input('Question about image','Describe this image.')
        if upload:
            if upload.size>8_000_000: st.error('Image limit is 8 MB')
            else:
                st.image(upload)
                if st.button('Analyze'): st.write(model.chat('llava',prompt,upload.getvalue()))
    elif mode=='Audio':
        upload=st.file_uploader('Audio',type=['wav','mp3','m4a','ogg'])
        if upload and st.button('Transcribe'):
            if upload.size>20_000_000: raise ValueError('Audio limit is 20 MB')
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/('audio'+Path(upload.name).suffix); p.write_bytes(upload.getvalue())
                segments,_=whisper_model().transcribe(str(p)); transcript=' '.join(s.text for s in segments)
            st.session_state['transcript']=transcript
        if 'transcript' in st.session_state:
            st.text_area('Transcript',st.session_state['transcript']); q=st.text_input('Question about transcript','Summarize the transcript.')
            if st.button('Ask Mistral'): st.write(model.chat('mistral',q+'\nTranscript:\n'+st.session_state['transcript']))
    else:
        upload=st.file_uploader('Text-based PDF',type=['pdf'])
        if upload:
            data=upload.getvalue(); digest=hashlib.sha256(data).hexdigest()
            if st.button('Index PDF'):
                records=pdf_records(data); st.session_state['index']=build_index(records,model); st.session_state['digest']=digest
                st.success(f'Indexed {len(records)} chunks.')
            question=st.text_input('Ask about the PDF')
            if st.button('Retrieve and answer') and question.strip():
                if st.session_state.get('digest')!=digest: st.warning('Index this PDF first.')
                else:
                    records=retrieve(st.session_state['index'],question,model); st.write(answer_pdf(model,question,records))
                    with st.expander('Retrieved evidence'):
                        for r in records: st.markdown(f"**Page {r['page']}**"); st.text(r['text'])
except Exception as e: st.error(f'Unable to complete request: {e}')
