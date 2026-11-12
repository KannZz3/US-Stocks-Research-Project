"""Session-owned upload controls. Uploaded frames never enter Streamlit's shared cache."""
import hashlib
import uuid
import streamlit as st
from upload_data import process,export_csv,eligibility


def render(uploaded):
    if uploaded is None:
        st.session_state.pop('upload_ready',None)
        return None
    content=uploaded.getvalue()
    key=hashlib.sha256(content).hexdigest()
    if st.session_state.get('upload_source')!=key:
        st.session_state['upload_source']=key
        st.session_state['adjusted_upload']=False
        st.session_state.pop('upload_ready',None)
        st.session_state['upload_training_cache']={}
    st.caption('Raw inputs: unadjusted OHLCV plus positive DlyCumFacShr. Engineered inputs: the complete 43-column schema, already split-adjusted.')
    confirm=st.checkbox('Engineered prices are already split-adjusted',key='adjusted_upload')
    if st.button('Validate / Process CSV',key='process_upload'):
        st.session_state.pop('upload_ready',None)
        try:
            with st.spinner('Validating and processing CSV'):
                st.session_state['upload_ready']=process(content,confirm)
        except (ValueError,AssertionError,OverflowError) as error:
            st.error('Upload rejected: '+(str(error) or 'A reference data-quality assertion failed.'))
    ready=st.session_state.get('upload_ready')
    if ready is None: return None
    data,audit=ready
    st.success(f"Valid CSV · {len(data):,} rows · {data.Ticker.nunique()} stocks")
    st.json(audit)
    st.dataframe(data.head(50),hide_index=True,width='stretch')
    st.download_button('Download engineered CSV',export_csv(data),'uploaded_sector.csv','text/csv')
    reasons=eligibility(data)
    for reason in reasons: st.warning(reason)
    if st.radio('Research dataset',['Built-in sectors','Uploaded dataset'],key='active_dataset')=='Uploaded dataset':
        st.session_state.setdefault('upload_scope',uuid.uuid4().hex)
        return data
    return None
